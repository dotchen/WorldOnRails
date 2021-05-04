import os
import math
import yaml
import lmdb
import numpy as np
import torch
import wandb
import carla
import random
import string

from torch.distributions.categorical import Categorical

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from utils import visualize_obs, _numpy

from rails.bellman import BellmanUpdater
from rails.models import EgoModel
from autoagents.waypointer import Waypointer

def get_entry_point():
    return 'QCollector'

FPS = 20.
STOP_THRESH = 0.1
MAX_STOP = 500

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

class QCollector(AutonomousAgent):

    """
    action value agent but assumes a static world
    """

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """

        self.track = Track.MAP
        self.num_frames = 0
        
        with open(path_to_conf_file, 'r') as f:
            config = yaml.safe_load(f)
        
        for key, value in config.items():
            setattr(self, key, value)
        
        device = torch.device('cuda')
        ego_model = EgoModel(1./FPS*(self.num_repeat+1)).to(device)
        ego_model.load_state_dict(torch.load(self.ego_model_dir))
        ego_model.eval()
        BellmanUpdater.setup(config, ego_model, device=device)
        
        self.vizs      = []
        self.wide_rgbs = []
        self.narr_rgbs = []
        self.wide_sems = []
        self.narr_sems = []

        self.lbls = []
        self.locs = []
        self.rots = []
        self.spds = []
        self.cmds = []

        self.waypointer = None

        if self.log_wandb:
            wandb.init(project='carla_data_phase1')

        self.noiser = OrnsteinUhlenbeckActionNoise(dt=1/FPS)
        self.prev_steer = 0
        
        self.stop_count = 0

    def destroy(self):
        if len(self.lbls) == 0:
            return

        self.flush_data()

    def flush_data(self):

        if self.log_wandb:
            wandb.log({
                'vid': wandb.Video(np.stack(self.vizs).transpose((0,3,1,2)), fps=20, format='mp4')
            })

        # Save data
        data_path = os.path.join(self.main_data_dir, _random_string())
        print ('Saving to {}'.format(data_path))

        lmdb_env = lmdb.open(data_path, map_size=int(1e10))

        length = len(self.lbls)
        with lmdb_env.begin(write=True) as txn:

            txn.put('len'.encode(), str(length).encode())

            for i in range(length):
                
                for idx in range(len(self.camera_yaws)):
                    txn.put(
                        f'wide_rgb_{idx}_{i:05d}'.encode(),
                        np.ascontiguousarray(self.wide_rgbs[i][idx]).astype(np.uint8),
                    )
    
                    txn.put(
                        f'narr_rgb_{idx}_{i:05d}'.encode(),
                        np.ascontiguousarray(self.narr_rgbs[i][idx]).astype(np.uint8),
                    )
    
                    txn.put(
                        f'wide_sem_{idx}_{i:05d}'.encode(),
                        np.ascontiguousarray(self.wide_sems[i][idx]).astype(np.uint8),
                    )
                    
                    txn.put(
                        f'narr_sem_{idx}_{i:05d}'.encode(),
                        np.ascontiguousarray(self.narr_sems[i][idx]).astype(np.uint8),
                    )

                txn.put(
                    f'lbl_{i:05d}'.encode(),
                    np.ascontiguousarray(self.lbls[i]).astype(np.uint8),
                )

                txn.put(
                    f'loc_{i:05d}'.encode(),
                    np.ascontiguousarray(self.locs[i]).astype(np.float32)
                )

                txn.put(
                    f'rot_{i:05d}'.encode(),
                    np.ascontiguousarray(self.rots[i]).astype(np.float32)
                )

                txn.put(
                    f'spd_{i:05d}'.encode(),
                    np.ascontiguousarray(self.spds[i]).astype(np.float32)
                )

                
                txn.put(
                    f'cmd_{i:05d}'.encode(),
                    np.ascontiguousarray(self.cmds[i]).astype(np.float32)
                )

        self.vizs.clear()
        self.wide_rgbs.clear()
        self.narr_rgbs.clear()
        self.wide_sems.clear()
        self.narr_sems.clear()
        self.lbls.clear()
        self.locs.clear()
        self.rots.clear()
        self.spds.clear()
        self.cmds.clear()
        
        lmdb_env.close()

    def sensors(self):
        sensors = [
            {'type': 'sensor.collision', 'id': 'COLLISION'},
            {'type': 'sensor.map', 'id': 'MAP'},
            {'type': 'sensor.speedometer', 'id': 'EGO'},
            {'type': 'sensor.other.gnss', 'x': 0., 'y': 0.0, 'z': self.camera_z, 'id': 'GPS'},
        ]
        
        # Add sensors
        for i, yaw in enumerate(self.camera_yaws):
            x = self.camera_x*math.cos(yaw*math.pi/180)
            y = self.camera_x*math.sin(yaw*math.pi/180)
            sensors.append({'type': 'sensor.stitch_camera.rgb', 'x': x, 'y': y, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': yaw,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'Wide_RGB_{i}'})
            sensors.append({'type': 'sensor.stitch_camera.semantic_segmentation', 'x': x, 'y': y, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': yaw,
            'width': 160, 'height': 240, 'fov': 60, 'id': f'Wide_SEG_{i}'})
            sensors.append({'type': 'sensor.camera.rgb', 'x': x, 'y': y, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': yaw,
            'width': 384, 'height': 240, 'fov': 50, 'id': f'Narrow_RGB_{i}'})
            sensors.append({'type': 'sensor.camera.semantic_segmentation', 'x': x, 'y': y, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': yaw,
            'width': 384, 'height': 240, 'fov': 50, 'id': f'Narrow_SEG_{i}'})
            
        return sensors

    def run_step(self, input_data, timestamp):
        
        wide_rgbs = []
        narr_rgbs = []
        wide_sems = []
        narr_sems = []

        for i in range(len(self.camera_yaws)):
            
            _, wide_rgb = input_data.get(f'Wide_RGB_{i}')
            _, narr_rgb = input_data.get(f'Narrow_RGB_{i}')
            _, wide_sem = input_data.get(f'Wide_SEG_{i}')
            _, narr_sem = input_data.get(f'Narrow_SEG_{i}')
            
            wide_rgbs.append(wide_rgb[...,:3])
            narr_rgbs.append(narr_rgb[...,:3])
            wide_sems.append(wide_sem)
            narr_sems.append(narr_sem)


        _, lbl = input_data.get('MAP')
        _, col = input_data.get('COLLISION')
        _, ego = input_data.get('EGO')
        _, gps = input_data.get('GPS')

        if self.waypointer is None:
            self.waypointer = Waypointer(self._global_plan, gps)
            _, _, cmd = self.waypointer.tick(gps)
        else:
            _, _, cmd = self.waypointer.tick(gps)

        yaw = ego.get('rot')[-1]
        spd = ego.get('spd')
        loc = ego.get('loc')

        delta_locs, delta_yaws, next_spds = BellmanUpdater.compute_table(yaw/180*math.pi)

        # Convert lbl to rew maps
        lbl_copy = lbl.copy()
        waypoint_rews, stop_rews, brak_rews, free = BellmanUpdater.get_reward(lbl_copy, [0,0], ref_yaw=yaw/180*math.pi)

        waypoint_rews = waypoint_rews[None].expand(self.num_plan, *waypoint_rews.shape)
        brak_rews = brak_rews[None].expand(self.num_plan, *brak_rews.shape)
        stop_rews = stop_rews[None].expand(self.num_plan, *stop_rews.shape)
        free = free[None].expand(self.num_plan, *free.shape)

        # If it is idle, make it LANE_FOLLOW
        cmd_value = cmd.value-1
        cmd_value = 3 if cmd_value < 0 else cmd_value

        action_values, _ = BellmanUpdater.get_action(
            delta_locs, delta_yaws, next_spds,
            waypoint_rews[...,cmd_value], brak_rews, stop_rews, free,
            torch.zeros((self.num_plan,2)).float().to(BellmanUpdater._device),
            extract=(
                torch.tensor([[0.,0.]]),  # location
                torch.tensor([0.]),       # yaw
                torch.tensor([spd]),      # spd
            )
        )
        action_values = action_values.squeeze(0)

        action = int(Categorical(logits=action_values/self.temperature).sample())
        # action = int(action_values.argmax())

        steer, throt, brake = map(float, BellmanUpdater._actions[action])
        
        if self.noise_collect:
            steer += self.noiser()

        if len(self.vizs) > self.num_per_flush:
            self.flush_data()

        rgb = np.concatenate([wide_rgbs[0], narr_rgbs[0]], axis=1)
        spd = ego.get('spd')
        self.vizs.append(visualize_obs(rgb, yaw/180*math.pi, (steer, throt, brake), spd, cmd=cmd.value, lbl=lbl_copy))

        if col:
            self.flush_data()
            raise Exception('Collector has collided!! Heading out :P')

        if spd < STOP_THRESH:
            self.stop_count += 1
        else:
            self.stop_count = 0
        
        if cmd_value in [4,5]:
            actual_steer = steer
        else:
            actual_steer = steer * 1.2

        self.prev_steer = actual_steer

        # Save data
        if self.num_frames % (self.num_repeat+1) == 0 and self.stop_count < MAX_STOP:
            # Note: basically, this is like fast-forwarding. should be okay tho as spd is 0.
            self.wide_rgbs.append(wide_rgbs)
            self.narr_rgbs.append(narr_rgbs)
            self.wide_sems.append(wide_sems)
            self.narr_sems.append(narr_sems)
            self.lbls.append(lbl)
            self.locs.append(loc)
            self.rots.append(yaw)
            self.spds.append(spd)
            self.cmds.append(cmd_value)
        
        self.num_frames += 1
        
        return carla.VehicleControl(steer=actual_steer, throttle=throt, brake=brake)


def _random_string(length=10):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))
