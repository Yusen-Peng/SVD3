import os
import torch
from typing import List, Dict, Optional
import random
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def collect_sintel_frames(root: str, split: str, folder: str) -> List[str]:
    """Return list of frame paths under e.g. training/clean/alley_1/*.png."""
    fdir = os.path.join(root, split, folder)
    frames = []

    # log the number of scenes
    scenes = sorted([d for d in os.listdir(fdir) if os.path.isdir(os.path.join(fdir, d))])
    num_scenes = len(scenes)
    print("🍑" * 20)
    print(f"Found {num_scenes} scenes for calibration!")

    if not os.path.isdir(fdir):
        return frames
    for scene in sorted(os.listdir(fdir)):
        scene_dir = os.path.join(fdir, scene)
        if not os.path.isdir(scene_dir):
            continue
        for fn in sorted(os.listdir(scene_dir)):
            if fn.lower().endswith(".png"):
                frames.append(os.path.join(scene_dir, fn))
    return frames


def collect_scannet_frames(root: str) -> List[str]:
    """
    Collect calibration frames from ScanNetV2 using only the 'color_90' subset.
    Expected structure:
      root/
        sceneXXXX_XX/
          color/
          color_90/
            frame_XXXX.jpg
    """
    frames = []

    if not os.path.isdir(root):
        print(f"[WARN] ScanNet root not found: {root}")
        return frames

    # List scene folders
    scenes = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])
    print("🍑" * 20)
    print(f"Found {len(scenes)} ScanNet scenes for calibration!")

    for scene in scenes:
        # <root>/sceneXXXX_XX/color_90
        color90_dir = os.path.join(root, scene, "color_90")
        if not os.path.isdir(color90_dir):
            # some scenes may lack color_90 → skip
            continue

        # Collect JPEG frames
        for fn in sorted(os.listdir(color90_dir)):
            if fn.lower().endswith(".jpg"):
                frames.append(os.path.join(color90_dir, fn))

    print(f"Collected {len(frames)} total ScanNet calibration frames.")
    return frames


def build_transform(image_size: int = 224, center_crop: bool = True):
    tr = [transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC)]
    if center_crop:
        tr.append(transforms.CenterCrop(image_size))
    tr += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(tr)


def Pi3_get_calib_train_data(
    root: str,
    nsamples: int = 256,
    batch_size: int = 8,
    image_size: int = 224,
    split: str = "training",
    seed: int = 3
) -> List[Dict[str, torch.Tensor]]:
    # collect all frames
    if root.endswith('sintel'):
        frames = collect_sintel_frames(root, split, "final")
        print(f"🍑there are {len(frames)} total frames from Sintel {split} set.")
    elif 'scannet' in root.lower():
        frames = collect_scannet_frames(root)
        print(f"🍑there are {len(frames)} total frames from ScanNet set.")
    else:
        raise NotImplementedError("This dataset is not supported yet.")

    # sample `nsamples` frames randomly
    random.seed(seed)
    torch.manual_seed(seed)
    chosen = random.sample(frames, nsamples) if len(frames) >= nsamples else random.choices(frames, k=nsamples)

    # # FIXME: ablation - use all frames
    # chosen = frames

    # image preprocessing + batching
    to_tensor = build_transform(image_size=image_size, center_crop=True)
    traindataset: List[Dict[str, torch.Tensor]] = []
    batch_imgs: Optional[List[torch.Tensor]] = None
    for idx, path in tqdm(enumerate(chosen), total=len(chosen), desc="Preparing calibration data"):
        x = to_tensor(Image.open(path).convert("RGB"))
        if batch_imgs is None:
            batch_imgs = []
        batch_imgs.append(x)

        full = (len(batch_imgs) == batch_size)
        last = (idx == len(chosen) - 1)
        if full or last:
            pixel_values = torch.stack(batch_imgs, dim=0)
            traindataset.append({"pixel_values": pixel_values})
            batch_imgs = None

    # save the dataset
    if root.endswith('sintel'):
        torch.save(traindataset, f"/data/wanghaoxuan/SVD_Pi3_cache/sintel_pi3_calib_nsamples{nsamples}_size{image_size}_seed{seed}.pt")
    elif 'scannet' in root.lower():
        torch.save(traindataset, f"/data/wanghaoxuan/SVD_Pi3_cache/scannet_pi3_calib_nsamples{nsamples}_size{image_size}_seed{seed}.pt")
    else:
        raise NotImplementedError("This dataset is not supported yet.")
    
    return traindataset
