import traceback

from flask import json
from local_datasets.base.easy_dataset import EasyDataset
from pi3.utils.geometry import depthmap_to_absolute_camera_coordinates
import numpy as np
import os
import PIL
import pi3.utils.cropping as cropping
import torchvision.transforms as tvf
from omegaconf import OmegaConf
from .transforms import *
import pandas as pd
from .utils import *

def _sanitize_depth(dm):
    dm = np.asarray(dm)
    if dm.dtype == np.uint16:           # mm → m
        dm = dm.astype(np.float32) / 1000.0
    else:
        dm = dm.astype(np.float32)
    dm[~np.isfinite(dm)] = 0.0
    return dm

def _inpaint_depth(dm, valid, iters=30):
    """Very fast 4-neighbor Jacobi smoothing on invalid pixels only."""
    dm = dm.copy()
    invalid = ~valid
    if invalid.sum() == 0:
        return dm
    # seed invalids with border-mean to start (avoid zeros)
    seed = np.nanmean(np.where(valid, dm, np.nan))
    if not np.isfinite(seed): seed = 1.5
    dm[invalid] = seed
    for _ in range(iters):
        # 4-neighbor average
        avg = (
            np.roll(dm, 1, 0) + np.roll(dm, -1, 0) +
            np.roll(dm, 1, 1) + np.roll(dm, -1, 1)
        ) * 0.25
        dm[invalid] = avg[invalid]
    return dm

def _depth_stats(dm):
    v = np.isfinite(dm) & (dm > 1e-6)
    return v.mean(), (float(dm[v].min()) if v.any() else 0.0), (float(dm[v].max()) if v.any() else 0.0)

def synthesize_depth_if_needed(
    view,
    scene_key=None,
    fixed_backup=3.0,         # meters, last-resort constant plane
    dataset_prior=None,       # dict-like with "global_median"
    scene_priors=None,        # dict mapping scene_key -> median depth
):
    """
    Ensures view['depthmap'] becomes usable. Returns (depthmap, was_synthesized: bool).
    """
    dm = _sanitize_depth(view['depthmap'])
    valid = (dm > 1e-6) & np.isfinite(dm)
    ratio, dmin, dmax = _depth_stats(dm)

    if ratio == 0.0:
        # Case 3: totally empty -> try scene prior, then global prior, else constant
        if scene_key and scene_priors and scene_key in scene_priors:
            median_d = scene_priors[scene_key]
        elif dataset_prior and ("global_median" in dataset_prior) and np.isfinite(dataset_prior["global_median"]):
            median_d = float(dataset_prior["global_median"])
        else:
            median_d = fixed_backup
        dm = np.full_like(dm, median_d, dtype=np.float32)
        return dm, True
    return dm.astype(np.float32), False


def _has_valid_depth(dm: np.ndarray, z_near: float = 1e-6) -> bool:
    dm = np.asarray(dm)
    if dm.dtype == np.uint16:
        dm = dm.astype(np.float32) / 1000.0
    else:
        dm = dm.astype(np.float32)
    dm[~np.isfinite(dm)] = 0.0
    return bool((dm > z_near).any())

class BaseDataset(EasyDataset):
    def __init__(
        self,
        seed=2024,
        resolution=None,            # (width, height) or list of (width, height) or list of int
        aug_crop=False,             # False or int, slightly scale the image a bit larger than the target resolution
        aug_focal=False,            # False or float in [0, 1]
        z_far=0,
        frame_num=2,
        transform=tvf.ToTensor(),
        cache_file=None,
        save_cache=False,
        mode='train',
        cache_name=None,
        max_refetch=3,
        random_sample_thres=0.1,
        shuffle=True,
        use_sparse_depth=False,
    ):
        super().__init__()
        self.frame_num = frame_num

        self.transform = transform

        self.shuffle = shuffle

        self.use_sparse_depth = use_sparse_depth

        self._rng = np.random.default_rng(seed)
        self._set_resolutions(resolution)

        self.aug_crop = aug_crop
        self.aug_focal = aug_focal

        self.z_far = z_far

        self.dataset_label = 'BaseDataset'

        self.save_cache = save_cache
        self.cache_loaded = False
        self.cache_name = cache_name
        if cache_file is not None:
            print(f'[BaseDataset] Loading cache from {cache_file}..')
            res = self.load_cache(cache_file)
            if res:
                self.cache_loaded = True
                print(f'[BaseDataset] Cache is loaded.')

        self.mode = mode
        self.max_refetch = max_refetch

        self.random_sample_thres = random_sample_thres  # default not to do that

    def convert_attributes(self):
        """
        Avoid memory leak caused by python list or python dict
        https://github.com/pytorch/pytorch/issues/13246
        """

        def _is_equivalent(original, converted):
            """
            Check if the converted data structure is equivalent to the original.
            """
            try:
                return original == converted
            except Exception:
                return False

        for attr_name in dir(self):
            if attr_name.startswith("__") or callable(getattr(self, attr_name)):
                continue
            
            attr_value = getattr(self, attr_name)
            
            if isinstance(attr_value, list):
                try:
                    converted_value = np.array(attr_value)
                    
                    if _is_equivalent(attr_value, converted_value.tolist()):
                        setattr(self, attr_name, converted_value)
                    else:
                        print(f"[{self.dataset_label}] <{attr_name}> conversion may not be equivalent, skipping.", flush=True)
                except ValueError as e:
                    print(f"[{self.dataset_label}] Error converting <{attr_name}>: {e}", flush=True)
            
            elif isinstance(attr_value, dict):
                try:
                    converted_value = pd.Series(attr_value)
                    if _is_equivalent(attr_value, converted_value.to_dict()):
                        setattr(self, attr_name, converted_value)
                    else:
                        print(f"[{self.dataset_label}] <{attr_name}> conversion may not be equivalent, skipping.", flush=True)
                except ValueError as e:
                    print(f"[{self.dataset_label}] Error converting <{attr_name}>: {e}", flush=True)

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, 'undefined resolution'
        if OmegaConf.is_config(resolutions):
            resolutions = OmegaConf.to_object(resolutions)

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                width, height = resolution
            assert isinstance(width, int), f'Bad type for {width=} {type(width)=}, should be int'
            assert isinstance(height, int), f'Bad type for {height=} {type(height)=}, should be int'
            # assert width >= height
            # self._resolutions.append((width, height))
            self._resolutions.append([width, height])

        self.num_resoluions = len(self._resolutions)

    def _crop_resize_if_necessary(self, image, depthmap, intrinsics, resolution, rng=None, info=None, normal=None, far_mask=None):
        """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W-cx)
        min_margin_y = min(cy, H-cy)
        assert min_margin_x > W/5, f'Bad principal point in view={info}'
        assert min_margin_y > H/5, f'Bad principal point in view={info}'
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)
        image, depthmap, intrinsics, normal, far_mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, normal=normal)

        # transpose the resolution if necessary
        W, H = image.size  # new size
        # NOTE: Here we don't care about portrait image.
        # assert resolution[0] >= resolution[1]
        # if H > 1.1*W:
        #     # image is portrait mode
        #     resolution = resolution[::-1]
        # elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1]:
        #     # image is square, so we chose (portrait, landscape) randomly
        #     if rng.integers(2):
        #         resolution = resolution[::-1]

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_focal:
            crop_scale = self.aug_focal + (1.0 - self.aug_focal) * np.random.beta(0.5, 0.5) # beta distribution, bi-modal
            image, depthmap, intrinsics, normal, far_mask = cropping.center_crop_image_depthmap(image, depthmap, intrinsics, crop_scale, normal=normal, far_mask=far_mask)

        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        image, depthmap, intrinsics, normal, far_mask = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution, normal=normal, far_mask=far_mask) # slightly scale the image a bit larger than the target resolution

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
        crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
        image, depthmap, intrinsics2, normal, far_mask = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox, normal=normal, far_mask=far_mask)

        other = [x for x in [normal, far_mask] if x is not None]
        return image, depthmap, intrinsics2, *other

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            if len(idx) == 3:
                idx, ar_idx, frame_num = idx
                self.frame_num = frame_num
            else:
                idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        resolution = self._resolutions[ar_idx]  # DO NOT CHANGE THIS
        error = None
        for _ in range(10):  # retry budget
            try:
                views = self._get_views(idx, resolution, self._rng)

                if self.shuffle:
                    self._rng.shuffle(views)

                processed = []
                for raw_v, view in enumerate(views):
                    # build a stable scene key
                    scene_key = (view.get('dataset', ''), view.get('label', ''))

                    # synthesize if needed (or inpaint holes)
                    dm_fixed, synthesized = synthesize_depth_if_needed(
                        view,
                        scene_key=scene_key,
                        fixed_backup=3.0
                    )
                    view['depthmap'] = dm_fixed
                    view['depth_source'] = 'synthetic' if synthesized else 'original'

                    # proceed as usual, strictly
                    assert 'pts3d' not in view, "pts3d should not be present yet"
                    view['idx'] = (idx, ar_idx, len(processed))
                    w, h = view['img'].size
                    view['true_shape'] = np.int32((h, w))
                    assert 'camera_intrinsics' in view
                    if 'camera_pose' not in view:
                        view['camera_pose'] = np.full((4,4), np.nan, dtype=np.float32)
                    else:
                        assert np.isfinite(view['camera_pose']).all(), f'NaN in camera pose for {view_name(view)}'
                    assert np.isfinite(view['depthmap']).all(), f'NaN in depthmap for {view_name(view)}'

                    view['z_far'] = self.z_far
                    pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
                    geom_valid = valid_mask & np.isfinite(pts3d).all(axis=-1)
                    if geom_valid.sum() == 0:
                        # If even the synthesized depth can't produce valid 3D, skip this view
                        continue

                    view['pts3d'] = pts3d
                    view['valid_mask'] = geom_valid
                    view['depthmap'][~geom_valid] = 0.0
                    if 'normal' not in view:
                        view['normal'] = None

                    processed.append(view)
                    if len(processed) == self.frame_num:
                        break

                # If not enough valid views, resample (continue)
                if len(processed) < self.frame_num:
                    continue

                # Optional: trim to exactly frame_num (in case _get_views returned >frame_num valids)
                processed = processed[:self.frame_num]

                # 4) apply transforms only to the finalized, valid views
                for view in processed:
                    view['img'] = self.transform(view['img'])

                # success
                return processed

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                if hasattr(self, 'this_views_info'):
                    print(f"Failed to load data from {self.dataset_label}-{idx} ({self.this_views_info}) "
                        f"for error {type(e).__name__}: {e}\n{tb}", flush=True)
                else:
                    print(f"Failed to load data from {self.dataset_label}-{idx} "
                        f"for error {type(e).__name__}: {e}\n{tb}", flush=True)
                idx = np.random.randint(0, len(self))
                error = e

        # exhausted retries
        raise error

    def load_cache(self, cache_file):
        try:
            data = np.load(cache_file, allow_pickle=True).item()
            # Step 2: 遍历字典中的每个键值对，并将其赋值为类实例的属性
            if isinstance(data, dict):
                for key, value in data.items():
                    setattr(self, key, value)
                return True
            else:
                print("Error: The npy file does not contain a dictionary.")
                return False
        except Exception as e:
            print(f"An error occurred while loading the cache: {e}")
            return False

    def _save_cache(self, keys, desc=None):
        if desc is None:
            save_path = f'data/dataset_cache/{self.dataset_label}_cache.npy'
        else:
            save_path = f'data/dataset_cache/{self.dataset_label}_{desc}_cache.npy'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_dict = {}
        for key in keys:
            save_dict[key] = getattr(self, key)
        
        np.save(save_path, save_dict)

        print(f'Saved cache to {save_path}.', flush=True)

