import carla
import xml.etree.ElementTree as ET

from leaderboard.envs.map_utils import *



def viz(args):

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Parse points
    tree = ET.parse(args.route)
    for route in tree.iter("route"):
        
        town = route.attrib['town']
    
        world = client.load_world(town)
        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()
    
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.lincoln.mkz2017'))
        vehicle_bp.set_attribute('role_name', 'hero')
        
        start_pose = random.choice(spawn_points)
    
        player = world.spawn_actor(vehicle_bp, start_pose)
        player.set_autopilot(False)

        # Setup pygame
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        display = pygame.display.set_mode((320, 320),0,32)
        pygame.display.flip()
        # Set map drawer module
        input_module = ModuleInput(MODULE_INPUT)
        hud_module = ModuleHUD(MODULE_HUD, 320, 320)
        world_module = ModuleWorld(MODULE_WORLD, client, world, carla_map, player)
    
        # Register Modules
        module_manager.register_module(world_module)
        module_manager.register_module(hud_module)
        module_manager.register_module(input_module)
        module_manager.start_modules()
        
        # Get map surface
        map_surface = world_module.map_image.big_map_surface
        map_image = np.swapaxes(pygame.surfarray.array3d(map_surface),0,1)
        print (map_image.shape)
        
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    
        f, ax = plt.subplots(1,1,figsize=(20,15))
        ax.imshow(map_image)
        
        
        for i, waypoint in enumerate(route.iter('waypoint')):
            location = carla.Location(x=float(waypoint.attrib['x']),
                                                    y=float(waypoint.attrib['y']),
                                                    z=float(waypoint.attrib['z']))

            pixel_x, pixel_y = world_module.map_image.world_to_pixel(location)
            circle = Circle((pixel_x,pixel_y),5,color='red')
            ax.add_patch(circle)
            ax.text(pixel_x,pixel_y,'%d'%i,fontsize=10,color='blue')

        break

    plt.savefig('%s.png'%args.route.split('/')[-1].split('.xml')[0])
    
    player.destroy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--route', required=True)
    
    args = parser.parse_args()
    viz(args)