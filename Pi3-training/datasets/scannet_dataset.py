import sys
sys.path.append('.')

from datasets.base.base_dataset import BaseDataset
import os
import numpy as np
import os.path as osp
from PIL import Image
from datasets.base.transforms import *
import json
from tqdm import tqdm

class ScannetDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=240,                    # 80
        **kwargs
    ):
        super().__init__(**kwargs)

        assert data_root is not None

        self.verbose = verbose
        self.dataset_label = 'ScanNet'
        mode = self.mode
        self.data_root = data_root
        self.max_distance = max_distance

        self.sequences = os.listdir(data_root)
        if mode == 'train':
            self.sequences = [seq for seq in self.sequences if int(seq.split('_')[0][5:]) <= 660]
        else:
            self.sequences = [seq for seq in self.sequences if int(seq.split('_')[0][5:]) > 660]

        if self.verbose:
            print(f'[{self.dataset_label}] Sequences of {self.dataset_label} dataset:', self.sequences)

        print(f'[{self.dataset_label}] Found %d unique videos in %s' % (len(self.sequences), data_root), flush=True)

        with open('data/scannet_invalid_list.json') as f:
            self.invalid_list = json.load(f)

        if not os.path.exists(f'data/dataset_cache/scannetmv_{self.mode}_cache.npy'):
            self.num_imgs = {}
            for seq in tqdm(self.sequences):
                rgb_path = os.path.join(data_root, seq, 'color')
                self.num_imgs[seq] = len(os.listdir(rgb_path))

            np.save(f'data/dataset_cache/scannetmv_{self.mode}_cache', self.num_imgs)
        else:
            self.num_imgs = np.load(f'data/dataset_cache/scannetmv_{self.mode}_cache.npy', allow_pickle=True).item()

    def __len__(self):
        return len(self.sequences)
                    
    def _get_views(self, index, resolution, rng):
        scene = self.sequences[index]
        num_imgs = self.num_imgs[scene]
        valid_idxs = [i for i in range(num_imgs) if i not in self.invalid_list[scene]]
        num_imgs = len(valid_idxs)

        if self.frame_num > 16 and rng.random() < self.random_sample_thres:
            all_keys = valid_idxs
            should_replace = len(all_keys) < self.frame_num
            idxs = list(rng.choice(all_keys, size=self.frame_num, replace=should_replace))
        else:
            idxs = [rng.integers(0, num_imgs)]

            max_distance = int(self.max_distance / 8 * self.frame_num)
            start_idx = max(0, idxs[-1] - max_distance)
            end_idx = min(num_imgs-1, start_idx + 2*max_distance)
            start_idx = max(0, end_idx - 2*max_distance)
            valid_indices = np.arange(start_idx, end_idx + 1)

            if rng.random() < 0.5:
                should_replace = len(valid_indices) < self.frame_num - 1
                idxs.extend(list(rng.choice(valid_indices, self.frame_num-1, replace=should_replace)))
                idxs = [valid_idxs[i] for i in idxs]
            else:
                ref_frame_val = idxs[0]
                num_additional_to_select = self.frame_num - 1
                additional_selected_values = []
                pool_for_others_values = [val for val in valid_indices]
                pool_for_others_values.sort() 

                should_replace_for_others = len(pool_for_others_values) < num_additional_to_select

                if not pool_for_others_values: 
                    if should_replace_for_others:
                        additional_selected_values = [ref_frame_val] * num_additional_to_select
                else:
                    if not should_replace_for_others and len(pool_for_others_values) >= num_additional_to_select:
                        strata = np.array_split(pool_for_others_values, num_additional_to_select+1)
                        for stratum in strata:
                            if len(stratum) > 0 and ref_frame_val not in stratum: 
                                additional_selected_values.append(rng.choice(stratum))
                    else:
                        additional_selected_values = list(rng.choice(
                            pool_for_others_values,
                            num_additional_to_select,
                            replace=(should_replace_for_others or (len(pool_for_others_values) < num_additional_to_select))
                        ))

                idxs = [ref_frame_val, *additional_selected_values]
                idxs = [valid_idxs[idx] for idx in idxs]


        self.this_views_info = dict(
            scene=scene,
            idxs=idxs,
        )

        base_path = os.path.join(self.data_root, scene)
        intrinsic_path = osp.join(base_path, 'intrinsic/intrinsic_depth.txt')
        with open(intrinsic_path, 'r') as f:
            intrinsic_text = f.read()
        intrinsic = np.array([float(x) for x in intrinsic_text.split()]).astype(np.float32).reshape(4, 4)[:3, :3]
        
        views = []
        for idx in idxs:
            impath = os.path.join(base_path, 'color', f'{idx}.jpg')
            disppath = os.path.join(base_path, 'depth', f'{idx}.png')
            annotation = os.path.join(base_path, 'pose', f'{idx}.txt')

            # load camera params
            with open(annotation, 'r') as f:
                camera_pose_text = f.read()
            camera_pose = np.array([float(x) for x in camera_pose_text.split()]).astype(np.float32).reshape(4, 4)
            assert np.isfinite(camera_pose).all(), 'Infinite in camera pose for view'
            assert ~np.isnan(camera_pose).any(), 'NaN in camera pose for view'

            rgb_image = np.array(Image.open(impath).resize((640, 480), resample=lanczos))

            depthmap = Image.open(disppath).astype(np.float32) / 1000.

            rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsic.copy(), resolution, rng=rng, info=impath)

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsic_.astype(np.float32),
                dataset=self.dataset_label,
                label=scene,
                instance=str(idx),
            ))
        return views

