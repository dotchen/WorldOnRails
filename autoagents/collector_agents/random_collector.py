import os
import yaml
import lmdb
import wandb
import carla
import random
import string
import numpy as np
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from utils import visualize_obs

def get_entry_point():
    return 'RandomCollector'

class RandomCollector(AutonomousAgent):

    """
    Random autonomous agent to train vehicle dynamics model
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
        
        # Used for action repeat
        self.prev_ctr = 0
        self.prev_act = None

        self.rgbs = []              # RGB
        self.locs = []              # (x, y, z) in UE4 coordinate
        self.rots = []              # (roll, pitch, yaw) 
        self.spds = []              # in m/s
        self.acts = []              # (steer, throttle, brake)

        # launch-up wandb
        if self.log_wandb:
            wandb.init(project='carla_data_phase0', config=config)
            
        self.brake_prob = 1./(self.num_steers*self.num_throts+1)
        
    
    def destroy(self):
        if self.num_frames == 0:
            return # The run has not even started yet

        del self.rgbs
        del self.locs
        del self.rots
        del self.spds
        del self.acts

    def flush_data(self):

        if self.log_wandb:
            wandb.log({
                'num_frames': len(self.rgbs),
                'vid': wandb.Video(np.stack(self.rgbs).transpose((0,3,1,2)), fps=20, format='mp4')
            })

        # Save data
        data_path = os.path.join(self.ego_data_dir, _random_string())
        print ('Saving to {}'.format(data_path))

        lmdb_env = lmdb.open(data_path)
        
        with lmdb_env.begin(write=True) as txn:

            txn.put('len'.encode(), str(len(self.locs)).encode())

            # TODO: Save to wandb
            for i, (loc, rot, spd, act) in enumerate(zip(self.locs, self.rots, self.spds, self.acts)):
                txn.put(
                    f'loc_{i:05d}'.encode(),
                    np.ascontiguousarray(loc).astype(np.float32)
                )
                
                txn.put(
                    f'rot_{i:05d}'.encode(),
                    np.ascontiguousarray(rot).astype(np.float32)
                )
                
                txn.put(
                    f'spd_{i:05d}'.encode(),
                    np.ascontiguousarray(spd).astype(np.float32)
                )
                
                txn.put(
                    f'act_{i:05d}'.encode(),
                    np.ascontiguousarray(act).astype(np.float32)
                )

    def sensors(self):
        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 300, 'height': 200, 'fov': 100, 'id': 'Center'},
            {'type': 'sensor.speedometer',  'reading_frequency': 20, 'id': 'EGO'},
            {'type': 'sensor.collision', 'id': 'COLLISION'},
        ]

        return sensors

    def run_step(self, input_data, timestamp):
        
        if self.prev_act and self.prev_ctr < self.num_repeat:
            self.prev_ctr += 1
            return self.prev_act
        
        _, rgb = input_data.get('Center')
        _, ego = input_data.get('EGO')
        _, col = input_data.get('COLLISION')
        
        # Abort if collided
        if col:
            self.flush_data()
            raise Exception('Collector has collided!! Heading out :P')
        
        loc = ego.get('loc')
        rot = ego.get('rot')
        spd = ego.get('spd')

        # Randomize control
        if np.random.random() < self.brake_prob:
            steer = 0.
            throt = 0.
            brake = 1.
        else:
            steer = np.random.uniform(-self.max_steers, self.max_steers)
            throt = np.random.uniform(0, self.max_throts)
            brake = 0.

        act = [steer, throt, brake]
        
        # Add data
        if self.num_frames > self.num_ignore_first:
            self.locs.append(loc)
            self.rots.append(rot)
            self.spds.append(spd)
            self.acts.append(act)

            if self.log_wandb:
                self.rgbs.append(visualize_obs(rgb[...,:3], rot[-1], act, spd))
        
        control = carla.VehicleControl(steer=steer, throttle=throt, brake=brake)
        
        self.num_frames += 1
        
        self.prev_act = control
        self.prev_ctr = 0
        
        return control


def _random_string(length=10):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))
