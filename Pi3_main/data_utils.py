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


def collect_bonn_frames(root: str) -> List[str]:
    frames: List[str] = []

    if not os.path.isdir(root):
        print(f"[WARN] Bonn root not found: {root}")
        return frames

    # Top-level sequences (folders)
    seqs = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])

    print("🍑" * 20)
    print(f"Found {len(seqs)} Bonn sequences under: {root}")

    found_rgb110 = 0
    for seq in seqs:
        seq_dir = os.path.join(root, seq)
        rgb110_dir = os.path.join(seq_dir, "rgb_110")

        if not os.path.isdir(rgb110_dir):
            continue

        found_rgb110 += 1

        # Collect PNG frames under rgb_110
        pngs = sorted([
            fn for fn in os.listdir(rgb110_dir)
            if fn.lower().endswith(".png")
        ])

        # # Log per-sequence counts (helps debug missing folders fast)
        # print(f"  - {seq}: rgb_110 frames = {len(pngs)}")

        for fn in pngs:
            frames.append(os.path.join(rgb110_dir, fn))

    print("🍑" * 20)
    print(f"Found {found_rgb110} sequences with rgb_110/")
    print(f"Collected {len(frames)} total Bonn rgb_110 frames.")
    return frames


def collect_kitti_frames(
    root: str,
    date: Optional[str] = None,          # e.g. "2011_09_26" or None for all dates
    camera: str = "image_02",            # "image_02" (left color) / "image_03" (right color)
) -> List[str]:
    
    frames: List[str] = []

    if not os.path.isdir(root):
        print(f"[WARN] KITTI root not found: {root}")
        return frames

    # pick dates
    if date is not None:
        dates = [date]
    else:
        dates = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])

    print("🍑" * 20)
    print(f"Found {len(dates)} KITTI date folders under: {root}")

    num_drives = 0
    num_cam_dirs = 0

    for d in dates:
        date_dir = os.path.join(root, d)
        if not os.path.isdir(date_dir):
            continue

        # drives like 2011_09_26_drive_0005_sync
        drives = sorted([
            x for x in os.listdir(date_dir)
            if os.path.isdir(os.path.join(date_dir, x)) and x.endswith("_sync")
        ])

        if drives:
            print(f"  - {d}: {len(drives)} drives")

        for drive in drives:
            num_drives += 1
            drive_dir = os.path.join(date_dir, drive)

            cam_data_dir = os.path.join(drive_dir, camera, "data")
            if not os.path.isdir(cam_data_dir):
                continue

            num_cam_dirs += 1

            pngs = sorted([
                fn for fn in os.listdir(cam_data_dir)
                if fn.lower().endswith(".png")
            ])

            # log per-drive counts (helps debugging)
            print(f"    * {drive}/{camera}/data: {len(pngs)} frames")

            for fn in pngs:
                frames.append(os.path.join(cam_data_dir, fn))

    print("🍑" * 20)
    print(f"Found {num_drives} drives total; {num_cam_dirs} had {camera}/data/")
    print(f"Collected {len(frames)} total KITTI frames ({camera}).")
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


import os, shutil
from pathlib import Path

def dump_chosen_images(
    chosen: List[str],
    out_dir: str,
    max_images: Optional[int] = None,
    prefix: bool = True,
):
    """
    Dump chosen images into a single directory (flat structure).

    - Saves a manifest.txt for exact reproducibility
    - Optionally prefixes filenames with index to avoid collisions
    """
    out_dir: Path = Path(out_dir)

    # 💣 nuke existing directory
    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    selected = chosen[:max_images] if max_images else chosen

    # Save manifest (exact paths used)
    manifest = out_dir / "manifest.txt"
    with open(manifest, "w") as f:
        for p in selected:
            f.write(p + "\n")

    # Copy images
    for i, src in enumerate(selected):
        src = Path(src)

        # Avoid name collisions (very important if datasets mix!)
        if prefix:
            dst_name = f"{i:06d}_{src.name}"
        else:
            dst_name = src.name

        dst = out_dir / dst_name
        if not dst.exists():
            shutil.copy2(src, dst)

    print(f"📦 Dumped {len(selected)} images into {out_dir}")

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
    
    elif root.endswith('diverse'):
        print(f"😋😋😋we are collecting diverse scenes!!😋😋😋")
        frames_sintel = collect_sintel_frames('/data/wanghaoxuan/yusen_stuff/sintel', split, "final")
        frames_scannet = collect_scannet_frames('/data/wanghaoxuan/yusen_stuff/scannetv2')
        frames_bonn = collect_bonn_frames('/data/wanghaoxuan/yusen_stuff/rgbd_bonn_dataset')
        frames_kitti = collect_kitti_frames('/data/wanghaoxuan/yusen_stuff/kitti', camera='image_02')


        frames_dict = {
            'sintel': frames_sintel,
            'scannet': frames_scannet,
            'bonn': frames_bonn,
            'kitti': frames_kitti
        }


    else:
        raise NotImplementedError("This dataset is not supported yet.")


    if root.endswith('diverse'):
        num_frames_per_dataset = nsamples // len(frames_dict)
        chosen = []
        for _, dataset_frames in frames_dict.items():
            random.seed(seed)
            torch.manual_seed(seed)
            if len(dataset_frames) >= num_frames_per_dataset:
                sampled = random.sample(dataset_frames, num_frames_per_dataset)
            else:
                sampled = random.choices(dataset_frames, k=num_frames_per_dataset)
            chosen.extend(sampled)

    else:
        # sample `nsamples` frames randomly
        random.seed(seed)
        torch.manual_seed(seed)
        chosen = random.sample(frames, nsamples) if len(frames) >= nsamples else random.choices(frames, k=nsamples)


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
        torch.save(traindataset, f"/data/wanghaoxuan/yusen_stuff/SVD_Pi3_cache/curation/sintel_pi3_calib_nsamples{nsamples}_size{image_size}_seed{seed}.pt")
    elif 'scannet' in root.lower():
        torch.save(traindataset, f"/data/wanghaoxuan/yusen_stuff/SVD_Pi3_cache/curation/scannet_pi3_calib_nsamples{nsamples}_size{image_size}_seed{seed}.pt")
    elif 'diverse' in root.lower():
        torch.save(traindataset, f"/data/wanghaoxuan/yusen_stuff/SVD_Pi3_cache/curation/diverse_pi3_calib_nsamples{nsamples}_size{image_size}_seed{seed}.pt")
    else:
        raise NotImplementedError("This dataset is not supported yet.")


    # # dump chosen images for reference
    # dump_dir = f"/data/wanghaoxuan/yusen_stuff/SVD_Pi3_cache/curation/diverse"
    # dump_chosen_images(chosen, dump_dir)
    return traindataset



if __name__ == "__main__":
    # the original dataset
    # root = "/data/wanghaoxuan/yusen_stuff/scannetv2"
    
    root = "diverse"
    Pi3_get_calib_train_data(root, nsamples=256, batch_size=8, image_size=224, split="training", seed=3)