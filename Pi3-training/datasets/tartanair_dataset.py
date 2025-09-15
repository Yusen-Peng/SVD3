import sys
sys.path.append('.')

from datasets.base.base_dataset import BaseDataset
import os
import numpy as np
import os.path as osp
import h5py
from utils.basic import seed_anything
from PIL import Image
from tqdm import tqdm
from datasets.base.transforms import *

def xyzqxqyqxqw_to_c2w(xyzqxqyqxqw):
    xyzqxqyqxqw = np.array(xyzqxqyqxqw, dtype=np.float32)
    #NOTE: we need to convert x_y_z coordinate system to z_x_y coordinate system
    z, x, y = xyzqxqyqxqw[:3]
    qz, qx, qy, qw = xyzqxqyqxqw[3:]
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    c2w[:3, 3] = np.array([x, y, z])
    return c2w

class TarTanAirDataset(BaseDataset):
    def __init__(
        self,
        data_root='ssd:s3://TartanAir',
        verbose=False,
        max_distance=24,
        seq_num=-1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.verbose = verbose
        self.dataset_label = 'TarTanAir'
        self.max_distance = max_distance
        self.data_root = data_root

        self.sequences = []
        for seq in os.listdir(data_root):
            names = os.listdir(os.path.join(data_root, seq, seq, 'Easy'))
            seq_ = [(seq, 'Easy', name) for name in names]
            self.sequences.extend(seq_)
            names = os.listdir(os.path.join(data_root, seq, seq, 'Hard'))
            seq_ = [(seq, 'Hard', name) for name in names]
            self.sequences.extend(seq_)

        if seq_num > 0:
            self.sequences = self.sequences[:seq_num]

        self.sequences = sorted(self.sequences)

        if self.verbose:
            print(f'[{self.dataset_label}] Sequences of {self.dataset_label} dataset:', self.sequences)

        print(f'[{self.dataset_label}] Found {len(self.sequences)} unique videos in {data_root}', flush=True)

        fx = 320.0  # focal length x
        fy = 320.0  # focal length y
        cx = 320.0  # optical center x
        cy = 240.0  # optical center y

        # width = 640
        # height = 480

        self.intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        self.num_imgs = {}
        for seq in self.sequences:
            rgb_path = os.path.join(data_root, seq[0], seq[0], seq[1], seq[2], 'image_left')
            self.num_imgs[seq] = len(os.listdir(rgb_path))

    def __len__(self):
        return len(self.sequences)
                    
    def _get_views(self, index, resolution, rng):
        scene = self.sequences[index]
        num_imgs = self.num_imgs[scene]

        if self.frame_num > 20 and rng.random() < self.random_sample_thres:
            should_replace = num_imgs < self.frame_num
            idxs = list(rng.choice(num_imgs, size=self.frame_num, replace=should_replace))
        else:
            idxs = [rng.integers(0, num_imgs)]

            # max_distance = 12 if scene in self.special_scenes else self.max_distance
            max_distance = int(self.max_distance / 8 * self.frame_num)
            start_idx = max(0, idxs[-1] - max_distance)
            end_idx = min(num_imgs-1, start_idx + 2*max_distance)
            start_idx = max(0, end_idx - 2*max_distance)
            valid_indices = np.arange(start_idx, end_idx + 1)

            should_replace = len(valid_indices) < self.frame_num - 1
            idxs.extend(list(rng.choice(valid_indices, self.frame_num-1, replace=should_replace)))

        self.this_views_info = dict(
            scene=scene,
            pairs=idxs,
        )

        cam_path = os.path.join(self.data_root, scene[0], scene[0], scene[1], scene[2], 'pose_left.txt')
        caminfo = np.loadtxt(cam_path)

        views = []
        for idx in idxs:
            impath = os.path.join(self.data_root, scene[0], scene[0], scene[1], scene[2], 'image_left', f'{idx:06d}_left.png')
            depthpath = os.path.join(self.data_root, scene[0], scene[0], scene[1], scene[2], 'depth_left', f'{idx:06d}_left_depth.npy')

            # load camera params
            camera_pose = np.array(xyzqxqyqxqw_to_c2w(caminfo[idx]), dtype=np.float32)

            # load image and depth
            rgb_image = np.array(Image.open(impath))

            depthmap = np.load(depthpath)
            depthmap[depthmap > 80] = -1

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, self.intrinsics, resolution, rng=rng, info=impath)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset=self.dataset_label,
                label=f'{scene[0]}_{scene[1]}_{scene[2]}',
                instance=str(idx),
            ))
        return views
    

