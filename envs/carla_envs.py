import carla
import random
import numpy as np
import math
import weakref
import time
import collections
from utils import WEATHERS, RoadOption, Failure
from utils.local_planner import LocalPlanner 
from utils.map_utils import Wrapper as map_utils
from baselines.lbc.autopilot import AutoPilotController

STOP_THRESHOLD = 0.5

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu=0, sigma=0.1, theta=0.1, dt=0.1, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal()
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class CarlaEnv():

    def __init__(self, config, town='Town01', weathers=[1,3,6,8], num_vehicles=70, num_pedestrians=150, port=2000, noisy=False):
        
        self.town = town
        self.weathers = weathers

        self.fps = config.fps
        self.frame_skip = config.frame_skip
        self.num_vehicles = num_vehicles
        self.num_pedestrians = num_pedestrians
        self.ego_vehicle_name = config.ego_vehicle
        self.col_threshold = config.col_threshold
        self.bad_frame_limit = config.bad_frame_limit
        self.autopilot = config.autopilot
        self.init_spd = 5.0
        self.success_dist = 5.0
        self.max_tick = 20000

        self.max_deviation = config.max_deviation
        self.desired_speed = config.desired_speed

        # Related to policy's output type
        self.supervision = config.supervision

        # Game data
        self.noiser = OrnsteinUhlenbeckActionNoise(sigma=0.1 if noisy else 0, dt=1./config.fps)
        self.start = -1
        self.target = -1
        self.tick = 0
        self.frame = 0
        self.bad_frame = 0
        self.ego_vehicle = None
        self.weather = None
        self.collided = False
        self.collision_type_id = None
        self.collided_frame = -1
        self.traffic_tracker = None
        self.autopiloter = None

        self.max_steer = config.max_steer
        self.max_throttle = config.max_throttle
        self.num_steers = config.num_steers
        self.num_throttles = config.num_throttles

        self.actors = collections.defaultdict(list)

        # Setup carla misc.
        self.client = carla.Client('localhost', port)
        self.client.set_timeout(30.0)

        self.world = self.client.load_world(self.town)
        self.map = self.world.get_map()

        self.tm = self.client.get_trafficmanager(port-1)
        self.tm_port = self.tm.get_port()

        self.command = RoadOption.LANEFOLLOW



    def init(self, task=None, weather=None):

        # Set synchronous mode
        settings = self.world.get_settings()
        self.tm.set_synchronous_mode(True)
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1./self.fps
        self.world.apply_settings(settings)
        
        self.blueprints = self.world.get_blueprint_library()
        
        # Spawn ego-vehicle
        self.spawn_ego(task)
        
        # Spawn traffics
        self.spawn_vehicles(self.num_vehicles)
        self.spawn_pedestrians(self.num_pedestrians)
        self.traffic_tracker = TrafficTracker(self.ego_vehicle, self.world)
        
        # Setup map
        map_utils.init(self.client, self.world, self.map, self.ego_vehicle)
        
        # Set weather
        self.set_weather(weather)
        
        # Set autopilot if needed
        if self.autopilot:
            self.autopiloter = AutoPilotController(self.world, self.ego_vehicle, map_utils)
            
        return self.ready()

    def reset(self, task=None, weather=None):
        self.terminate()
        self.init(task=task, weather=weather)
        self.noiser.reset()
        
        obs, rew, done, info = self._step()
        return obs

    def step(self, action=None):
        for i in range(self.frame_skip):
            output = self._step(action)
        
        return output
    
    def _step(self, action=None):

        self.world.tick()
        map_utils.tick()

        self.traffic_tracker.tick()
        # Tick local planner
        self.local_planner.run_step()
        self.curr, self.command = self.local_planner.checkpoint
        self.next = self.local_planner.target[0].transform

        map_obs = map_utils.get_observations()
        labels = get_birdview(map_obs)

        loc = map_obs.get('position')
        ori = map_obs.get('orientation')
        spd = np.linalg.norm(map_obs.get('velocity'))
        har = self.is_hazard()
        if har:
            self.bad_frame = 0
            self.frame -= 1
        elif spd >= STOP_THRESHOLD:
            self.bad_frame = 0
        elif action is not None and spd < STOP_THRESHOLD:
            self.bad_frame += 1

        self.tick += 1
        self.frame += 1
        
        # Stitch camera obs
        rgb = np.concatenate([self.rgb_left,self.rgb_center,self.rgb_right],axis=1)
        sem = np.concatenate([self.sem_left,self.sem_center,self.sem_right],axis=1)

        obs = dict(
            rgb=rgb,
            sem=sem,
            lbl=labels, 
            loc=loc*map_utils.pixels_per_meter, 
            ori=ori,
            cmd=int(self.command)-1,
        )

        dis = distance(
            [self.curr.transform.location.x, self.curr.transform.location.y],
            [self.next.location.x, self.next.location.y],
            loc,
        )
        yaw = self.curr.transform.rotation.yaw

        success = self.is_success()
        failure = self.is_failure() or dis >= self.max_deviation
        done = success or failure
        
        if success:
            # Check termination condition
            terminate_condition = Failure.SUCCESS
        elif failure:
            if self.collision_type_id and self.collision_type_id.startswith('vehicle'):
                terminate_condition = Failure.COLLISION_BY_VEHICLES
            elif self.collision_type_id and self.collision_type_id.startswith('walker'):
                terminate_condition = Failure.COLLISION_BY_PEDS
            elif self.collision_type_id:
                terminate_condition = Failure.COLLISION_BY_OTHERS
            else:
                terminate_condition = Failure.DEVIATION
        else:
            terminate_condition = None

        reward = -10 if failure else self.reward(dis, yaw, spd, har)

        if action is None:
            pass
        elif self.autopilot:
            control = self.autopiloter()
            self.ego_vehicle.apply_control(control)
        elif isinstance(action, tuple):
            steer, throttle, brake = action
            control = carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)
            self.ego_vehicle.apply_control(control)
        else:
            control = self.get_control(action)
            self.ego_vehicle.apply_control(control)

        return obs, reward, done, dict(
            weather=self.weather,
            start=self.start,
            target=self.target,
            control=(control.throttle,control.steer,control.brake) if action is not None else (0,0,0),
            speed=spd,
            frame=self.frame,
            success=success,
            lights_ran=self.traffic_tracker.total_lights_ran,
            terminate_condition=terminate_condition,
        )

    def reward(self, dis, yaw, spd, har):
        ori_dis = abs((yaw - self.ego_vehicle.get_transform().rotation.yaw + 180)%360 - 180)

        desired_spd = 0 if har else self.desired_speed

        loc_reward = -min(dis, self.max_deviation)/self.max_deviation
        spd_reward =  min(1/(np.abs(spd - desired_spd)+1e-5), 1)
        ori_reward =  min(1/(ori_dis+1e-5), 1)

        reward = loc_reward + spd_reward + ori_reward

        return reward


    def get_control(self, action):
        if action == self.num_steers * self.num_throttles:
            control= carla.VehicleControl(steer=0., throttle=0., brake=1.)
        else:
            throttle = action // self.num_steers
            steer = action % self.num_steers

            control = carla.VehicleControl(
                steer=(2/(self.num_steers-1)*steer-1)*self.max_steer,
                throttle=1/(self.num_throttles-1)*throttle*self.max_throttle,
                brake=0.
            )
        
        
        control.steer += self.noiser()
        
        return control

    def terminate(self):
        self.tm.set_synchronous_mode(False)
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        
        for ped in self.actors['ped_controllers']:
            ped.stop()
        
        for sensor in self.actors['sensor']:
            sensor.destroy()
        
        for actor_type in self.actors.keys():
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actors[actor_type]])
            self.actors[actor_type].clear()
        
        self.ego_vehicle.destroy()
        self.ego_vehicle = None

        self.tick = 0
        self.frame = 0
        self.bad_frame = 0
        self.collided = False
        self.collision_type_id = None
        self.start = -1
        self.target = -1
        self.traffic_tracker = None
        self.autopiloter = None

        time.sleep(0.5)


    def set_weather(self, weather_id=None):
        if weather_id is None:
            weather_id = np.random.choice(self.weathers)
        self.weather = weather_id
        self.world.set_weather(WEATHERS[weather_id])

    def spawn_ego(self, task):
        blueprint = self.blueprints.find(self.ego_vehicle_name)
        blueprint.set_attribute('role_name', 'hero')

        spawn_points = self.map.get_spawn_points()
        
        if task is None:
            start, target = np.random.randint(len(spawn_points), size=2)
        else:
            start, target = task
            
        self.start = start
        self.target = target

        self.start_pose = spawn_points[start]
        self.target_pose = spawn_points[target]

        waypoint = self.map.get_waypoint(self.start_pose.location)

        self.ego_vehicle = self.world.spawn_actor(
            blueprint,
            self.start_pose
        )
        
        self.ego_vehicle.set_autopilot(False, self.tm_port)
        
        for name, yaw in {'left':-60,'center':0,'right':60}.items():
            rgb_camera_bp = self.blueprints.find('sensor.camera.rgb')
            rgb_camera_bp.set_attribute('image_size_x', '128')
            rgb_camera_bp.set_attribute('image_size_y', '160')
            rgb_camera_bp.set_attribute('fov', '60')
            rgb_camera = self.world.spawn_actor(
                rgb_camera_bp,
                carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0, yaw=yaw)),
                attach_to=self.ego_vehicle
            )
            
            sem_camera_bp = self.blueprints.find('sensor.camera.semantic_segmentation')
            sem_camera_bp.set_attribute('image_size_x', '128')
            sem_camera_bp.set_attribute('image_size_y', '160')
            sem_camera_bp.set_attribute('fov', '60')
            sem_camera = self.world.spawn_actor(
                sem_camera_bp,
                carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0, yaw=yaw)),
                attach_to=self.ego_vehicle
            )
            
            self.actors['sensor'].append(rgb_camera)
            self.actors['sensor'].append(sem_camera)
            
            rgb_camera.listen(self.__class__.get_on_camera(weakref.ref(self), f'rgb_{name}'))
            sem_camera.listen(self.__class__.get_on_camera(weakref.ref(self), f'sem_{name}'))

        # Attach collision sensor
        collision_sensor = self.world.spawn_actor(
            self.blueprints.find('sensor.other.collision'),
            carla.Transform(),
            attach_to=self.ego_vehicle
        )
        self.actors['sensor'].append(collision_sensor)

        collision_sensor.listen(lambda event: self.__class__.on_collision(weakref.ref(self), event))


    def spawn_vehicles(self, num_vehicles):

        blueprints = self.blueprints.filter('vehicle.*')
        spawn_points = np.random.choice(self.world.get_map().get_spawn_points(), num_vehicles)
        batch = list()

        for i, transform in enumerate(spawn_points):
            bp = np.random.choice(blueprints)
            batch.append(
                carla.command.SpawnActor(bp, transform).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, self.tm_port))
            )

        vehicles = list()
        errors = set()

        for msg in self.client.apply_batch_sync(batch):
            if msg.error:
                errors.add(msg.error)
            else:
                vehicles.append(msg.actor_id)

        if errors:
            print('\n'.join(errors))

        print('%d / %d vehicles spawned.' % (len(vehicles), num_vehicles))

        self.actors['vehicles'] = vehicles


    def spawn_pedestrians(self, num_pedestrians):

        blueprints = self.blueprints.filter('walker.pedestrian.*')
        spawn_points = []
        while len(spawn_points) < num_pedestrians:
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)

        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = np.random.choice(blueprints)
            batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

        pedestrians = list()
        ped_controllers = list()
        errors = set()

        for msg in self.client.apply_batch_sync(batch):
            if msg.error:
                errors.add(msg.error)
            else:
                pedestrians.append(msg.actor_id)

        if errors:
            print('\n'.join(errors))

        batch = []
        blueprint = self.world.get_blueprint_library().find('controller.ai.walker')
        for pedestrian in pedestrians:
            batch.append(
                carla.command.SpawnActor(blueprint, carla.Transform(), pedestrian)
            )

        results = self.client.apply_batch_sync(batch, True)
        for i, msg in enumerate(self.client.apply_batch_sync(batch)):
            if msg.error:
                errors.add(msg.error)
            else:
                ped_controllers.append(msg.actor_id)
        
        print('%d / %d pedestrians spawned.' % (len(ped_controllers), num_pedestrians))
        self.world.set_pedestrians_cross_factor(0.2)
        
        self.actors['pedestrians'].extend(pedestrians)
        self.actors['ped_controllers'].extend(self.world.get_actors(ped_controllers))
    
    def ready(self, ticks=50):
        
        # Set planner
        self.local_planner = LocalPlanner(self.ego_vehicle, 2.5, 3.0, 3.0)
        self.local_planner.set_route(self.start_pose.location, self.target_pose.location)
        self.timeout = self.local_planner.calculate_timeout(self.fps)
        if self.autopilot:
            self.autopiloter.set_route(self.start_pose.location, self.target_pose.location)
        
        # Set pedestrians
        for ped in self.actors['ped_controllers']:
            ped.start()
            ped.go_to_location(self.world.get_random_location_from_navigation())
            ped.set_max_speed(1 + random.random())
        
        for _ in range(ticks):
            output = self._step()
            
        # Set initial velocity
        ori = self.ego_vehicle.get_transform().get_forward_vector()
        self.ego_vehicle.set_velocity(ori*self.init_spd)
        
        return output
    
    @staticmethod
    def rgb_to_array(event):
        array = np.frombuffer(event.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (event.height, event.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        
        return array
        
    @staticmethod
    def get_on_camera(weakself, name):
        def on_camera(event):
            self = weakself()
            array = CarlaEnv.rgb_to_array(event)
            
            if 'sem' in name: 
                setattr(self, name, array[:, :, 0])
            else:
                setattr(self, name, array)
            
        return on_camera
        
    @staticmethod
    def on_collision(weakself, event):
        self = weakself()
        if not self:
            return
        
        impulse = event.normal_impulse
        intensity = np.linalg.norm([impulse.x, impulse.y, impulse.z])
        
        if intensity > self.col_threshold and event.frame >= 50:
            self.collided = True
            self.collision_type_id = event.other_actor.type_id
            
    
    def is_failure(self):
        location = self.ego_vehicle.get_location()
        if self.frame >= self.timeout or self.tick >= self.max_tick:
            return True
        elif self.collided:
            return True
        elif self.bad_frame >= self.bad_frame_limit:
            return True

        return False

    def is_success(self):
        location = self.ego_vehicle.get_location()
        distance = location.distance(self.target_pose.location)

        return distance <= self.success_dist

    def is_hazard(self):
        actors = self.world.get_actors()
        vehicles = actors.filter('*vehicle*')
        walkers = actors.filter('*walker*')
        lights = actors.filter('*traffic_light*')

        ego_vehicle_location = self.ego_vehicle.get_location()
        ego_vehicle_waypoint = self.map.get_waypoint(ego_vehicle_location)

        for target_vehicle in vehicles:
            # do not account for the ego vehicle
            if target_vehicle.id == self.ego_vehicle.id:
                continue

            # if the object is not in our lane it's not an obstacle
            target_vehicle_waypoint = self.map.get_waypoint(target_vehicle.get_location())
            if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
                    target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue

            if is_within_distance_ahead(target_vehicle.get_transform().location, self.ego_vehicle.get_transform().location,
                                        self.ego_vehicle.get_transform().rotation.yaw,
                                        10.):
                return True

        for walker in walkers:
            loc = walker.get_location()
            dist = loc.distance(ego_vehicle_location)
            degree = 162 / (np.clip(dist, 1.5, 10.5)+0.3)

            if map_utils.is_point_on_sidewalk(loc):
                continue

            if is_within_distance_ahead(loc, ego_vehicle_location,
                                        self.ego_vehicle.get_transform().rotation.yaw,
                                        10., degree=degree):

                return True

        for traffic_light in lights:
            object_location = self._get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self.map.get_waypoint(object_location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if is_within_distance_ahead(object_waypoint.transform.location, self.ego_vehicle.get_transform().location,
                                        self.ego_vehicle.get_transform().rotation.yaw,
                                        10.):
                if traffic_light.state == carla.TrafficLightState.Red:
                    return True

        return False

    def _get_trafficlight_trigger_location(self, traffic_light):  # pylint: disable=no-self-use
        """
        Calculates the yaw of the waypoint that represents the trigger volume of the traffic light
        """
        def rotate_point(point, radians):
            """
            rotate a given point by a given angle
            """
            rotated_x = math.cos(radians) * point.x - math.sin(radians) * point.y
            rotated_y = math.sin(radians) * point.x - math.cos(radians) * point.y

            return carla.Vector3D(rotated_x, rotated_y, point.z)

        base_transform = traffic_light.get_transform()
        base_rot = base_transform.rotation.yaw
        area_loc = base_transform.transform(traffic_light.trigger_volume.location)
        area_ext = traffic_light.trigger_volume.extent

        point = rotate_point(carla.Vector3D(0, 0, area_ext.z), math.radians(base_rot))
        point_location = area_loc + carla.Location(x=point.x, y=point.y)

        return carla.Location(point_location.x, point_location.y, point_location.z)


class TrafficTracker(object):
    LANE_WIDTH = 5.0

    def __init__(self, agent, world):
        self._agent = agent
        self._world = world

        self._prev = None
        self._cur = None

        self.total_lights_ran = 0
        self.total_lights = 0
        self.ran_light = False
        
        self.last_light_id = -1

    def tick(self):
        self.ran_light = False
        self._prev = self._cur
        self._cur = self._agent.get_location()

        if self._prev is None or self._cur is None:
            return

        light = TrafficTracker.get_active_light(self._agent, self._world)
        active_light = light
        
        if light is not None and light.id != self.last_light_id:
            self.total_lights += 1
            self.last_light_id = light.id
            
        light = TrafficTracker.get_closest_light(self._agent, self._world)

        if light is None or light.state != carla.libcarla.TrafficLightState.Red:
            self.active_red = False
            return

        light_location = light.get_transform().location
        light_orientation = light.get_transform().get_forward_vector()

        delta = self._cur - self._prev

        p = np.array([self._prev.x, self._prev.y])
        r = np.array([delta.x, delta.y])

        q = np.array([light_location.x, light_location.y])
        s = TrafficTracker.LANE_WIDTH * np.array([-light_orientation.x, -light_orientation.y])

        if TrafficTracker.line_line_intersect(p, r, q, s):
            self.ran_light = True
            self.total_lights_ran += 1


    @staticmethod
    def get_closest_light(agent, world):
        location = agent.get_location()
        closest = None
        closest_distance = float('inf')

        for light in world.get_actors().filter('*traffic_light*'):
            delta = location - light.get_transform().location
            distance = np.sqrt(sum([delta.x ** 2, delta.y ** 2, delta.z ** 2]))

            if distance < closest_distance:
                closest = light
                closest_distance = distance

        return closest

    @staticmethod
    def get_active_light(ego_vehicle, world):
        
        _map = world.get_map()
        ego_vehicle_location = ego_vehicle.get_location()
        ego_vehicle_waypoint = _map.get_waypoint(ego_vehicle_location)
        
        lights_list = world.get_actors().filter('*traffic_light*')
        
        for traffic_light in lights_list:
            location = traffic_light.get_location()
            object_waypoint = _map.get_waypoint(location)
            
            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue
            if object_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
                continue
            
            if not is_within_distance_ahead(
                    location,
                    ego_vehicle_location,
                    ego_vehicle.get_transform().rotation.yaw,
                    10., degree=60):
                continue

            return traffic_light

        return None

    @staticmethod
    def line_line_intersect(p, r, q, s):
        r_cross_s = np.cross(r, s)
        q_minus_p = q - p

        if abs(r_cross_s) < 1e-3:
            return False

        t = np.cross(q_minus_p, s) / r_cross_s
        u = np.cross(q_minus_p, r) / r_cross_s

        if t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0:
            return True

        return False

def is_within_distance_ahead(target_location, current_location, orientation, max_distance, degree=60):
    u = np.array([
        target_location.x - current_location.x,
        target_location.y - current_location.y])
    distance = np.linalg.norm(u)

    if distance > max_distance:
        return False

    v = np.array([
        math.cos(math.radians(orientation)),
        math.sin(math.radians(orientation))])

    angle = math.degrees(math.acos(np.dot(u, v) / (distance+1e-5)))

    return angle < degree

def distance(p1, p2, p3):
    p1, p2, p3 = map(np.array, [p1, p2, p3])
    return np.abs(np.cross(p2-p1, p1-p3))/(np.linalg.norm(p2-p1)+1e-5)


def get_birdview(observations):
    birdview = [
            observations['road'],
            observations['lane'],
            observations['traffic'],
            observations['vehicle'],
            observations['pedestrian'],
            observations['waypoints'][0],
            observations['waypoints'][1],
            observations['waypoints'][2],
            observations['waypoints'][3],
            observations['waypoints'][4],
            ]

    birdview = [x if x.ndim == 3 else x[...,None] for x in birdview]
    birdview = np.concatenate(birdview, 2)

    return birdview
