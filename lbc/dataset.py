import lmdb
import glob
import yaml
import math
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from common.augmenter import augment
from utils import filter_sem

MAP_SIZE = 96
PIXELS_PER_METER = 3

class LBCDataset(Dataset):
    def __init__(self, data_dir, config_path, jitter=False):
        super().__init__()
        self.augmenter = augment(0.5)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.crop_size = 64
        self.margin = (MAP_SIZE - self.crop_size)//2

        self.T = config['num_plan']
        self.x_jitter  = config['x_jitter'] if jitter else 0
        self.a_jitter  = config['a_jitter'] if jitter else 0
        self.seg_channels = config['seg_channels']
        self.camera_yaws = config['camera_yaws']
        self.crop_top = config['crop_top']
        self.crop_bottom = config['crop_bottom']
        
        self.num_frames = 0
        self.txn_map = dict()
        self.idx_map = dict()
        self.yaw_map = dict()
        self.file_map = dict()

        # Load dataset
        for full_path in glob.glob(f'{data_dir}/**'):
            txn = lmdb.open(
                full_path,
                max_readers=1, readonly=True,
                lock=False, readahead=False, meminit=False).begin(write=False)
            
            n = int(txn.get('len'.encode()))
            if n < self.T+1:
                print (full_path, ' is too small. consider deleting it.')
                txn.__exit__()
            else:
                offset = self.num_frames
                for i in range(n-self.T):
                    self.num_frames += 1
                    for j in range(len(self.camera_yaws)):
                        self.txn_map[(offset+i)*len(self.camera_yaws)+j] = txn
                        self.idx_map[(offset+i)*len(self.camera_yaws)+j] = i
                        self.yaw_map[(offset+i)*len(self.camera_yaws)+j] = j
                        self.file_map[(offset+i)*len(self.camera_yaws)+j] = full_path

        print(f'{data_dir}: {self.num_frames} frames (x{len(self.camera_yaws)})')

    def __len__(self):
        return self.num_frames*len(self.camera_yaws)

    def __getitem__(self, idx):

        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]
        cam_index = self.yaw_map[idx]

        locs = self.__class__.access('loc', lmdb_txn, index, self.T+1, dtype=np.float32)[:,:2]
        rot = self.__class__.access('rot', lmdb_txn, index, 1, dtype=np.float32)
        spd = self.__class__.access('spd', lmdb_txn, index, 1, dtype=np.float32)
        lbl = self.__class__.access('lbl', lmdb_txn, index, 1, dtype=np.uint8).reshape(96,96,12)
        cmd = self.__class__.access('cmd', lmdb_txn, index, 1, dtype=np.float32).flatten()

        rgb = self.__class__.access('wide_rgb_{}'.format(cam_index),  lmdb_txn, index, 1, dtype=np.uint8).reshape(240,480,3)
        sem = self.__class__.access('wide_sem_{}'.format(cam_index),  lmdb_txn, index, 1, dtype=np.uint8).reshape(240,480)

        yaw = float(rot)
        spd = float(spd)

        # Jitter
        x_jitter = np.random.randint(-self.x_jitter,self.x_jitter+1)
        a_jitter = np.random.randint(-self.a_jitter,self.a_jitter+1) + self.camera_yaws[cam_index]

        # Rotate BEV
        lbl = rotate_image(lbl, a_jitter+yaw+90)
        lbl = lbl[self.margin:self.margin+self.crop_size,self.margin+x_jitter:self.margin+x_jitter+self.crop_size]

        # Rotate locs
        dloc = rotate_points(locs[1:] - locs[0:1], -a_jitter-yaw-90)*PIXELS_PER_METER - [x_jitter-self.crop_size/2,-self.crop_size/2]

        # Augment RGB
        rgb = self.augmenter(images=rgb[None,...,::-1])[0]
        sem = filter_sem(sem, labels=self.seg_channels)

        rgb = rgb[self.crop_top:-self.crop_bottom]
        sem = sem[self.crop_top:-self.crop_bottom]

        return rgb, lbl, sem, dloc, spd, int(cmd)
        
    @staticmethod
    def access(tag, lmdb_txn, index, T, dtype=np.float32):
        return np.stack([np.frombuffer(lmdb_txn.get((f'{tag}_{t:05d}').encode()), dtype) for t in range(index,index+T)])




def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result
    
def rotate_points(points, angle):
    radian = angle * math.pi/180
    return points @ np.array([[math.cos(radian), math.sin(radian)], [-math.sin(radian), math.cos(radian)]])


def data_loader(memory, batch_size):
    return DataLoader(memory, batch_size=batch_size, num_workers=8, drop_last=True, shuffle=True)
