import math
import torch
import torch.nn.functional as F
import torchvision.transforms as tvf
import time
import torch.nn as nn
from typing import List, Optional, Tuple
from omegaconf import DictConfig
from PIL import Image

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from Pi3_main.pi3.models.pi3 import Pi3
from Pi3_main.pi3.utils.geometry import se3_inverse


class TwoFactorLinear(nn.Module):
    def __init__(self, in_features, out_features, r, has_bias):
        super().__init__()
        self.v = nn.Linear(in_features, r, bias=False)
        self.u = nn.Linear(r, out_features, bias=has_bias)
    def forward(self, x):
        # order matters: x -> V -> U  (reconstructs W = U S V^T)
        return self.u(self.v(x))

# Which leaves we factorized
_FACTOR_LEAVES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")

def install_twofactor_modules_from_sd(model: Pi3, sd):
    """
    For each target Linear in Pi3, if sd has <leaf>.u.weight and <leaf>.v.weight,
    replace that module with a TwoFactorLinear of the correct rank/bias so that
    state_dict keys match and load cleanly.
    """
    for i, blk in enumerate(model.decoder):
        for leaf in _FACTOR_LEAVES:
            base = f"decoder.{i}.{leaf}"
            k_u_w = f"{base}.u.weight"
            k_v_w = f"{base}.v.weight"
            k_u_b = f"{base}.u.bias"
            if (k_u_w in sd) and (k_v_w in sd):
                # Walk to parent module that owns the leaf
                parent = blk
                parts = leaf.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                leaf_name = parts[-1]
                old: nn.Linear = getattr(parent, leaf_name) # original nn.Linear

                in_f, out_f = old.in_features, old.out_features
                r = sd[k_v_w].shape[0]
                has_bias = (k_u_b in sd)

                # Build TwoFactorLinear with correct geometry
                tfl = TwoFactorLinear(in_features=in_f, out_features=out_f, r=r, has_bias=has_bias)
                tfl = tfl.to(device=old.weight.device, dtype=old.weight.dtype)

                setattr(parent, leaf_name, tfl)
    return model

class SlicableTwoFactorLinear(nn.Module):
    """
    y = (x @ V^T) @ U^T + b
    where:
      V: (r_max, in_features)
      U: (out_features, r_max)
    """
    def __init__(self, in_features, out_features, r_max, bias: bool):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r_max = r_max

        self.V = nn.Parameter(torch.empty(r_max, in_features))      # (r, in)
        self.U = nn.Parameter(torch.empty(out_features, r_max))     # (out, r)
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        # harmless - will be overwritten by load_state_dict anyway
        nn.init.normal_(self.V, std=0.02)
        nn.init.normal_(self.U, std=0.02)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def set_active_frac(self, frac: float | None):
        """
            Set the active ran fraction for this layer to support dynamic slicing.
        """
        self.active_frac = None if frac is None else float(frac)

    def forward(self, x, r: int | None = None):
        if r is None:
            frac = getattr(self, "active_frac", None)
            if frac is not None:
                r = int(round(self.r_max * frac))
        if r is None:
            r = self.r_max
        r = max(1, min(int(r), self.r_max))

        # z = x @ V[:r].T  -> (..., r)
        z = x.matmul(self.V[:r].T)
        # y = z @ U[:, :r].T -> (..., out)
        y = z.matmul(self.U[:, :r].T)
        if self.bias is not None:
            y = y + self.bias
        return y

def _get_module_by_dotted(root, dotted: str):
    cur = root
    for p in dotted.split("."):
        cur = getattr(cur, p)
    return cur


def _set_module_by_dotted(root, dotted: str, new_mod: nn.Module):
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


@torch.no_grad()
def install_slicabletwofactor_modules_from_sd(model: nn.Module, sd: dict):
    for i, blk in enumerate(model.decoder):
        for leaf in _FACTOR_LEAVES:
            base = f"decoder.{i}.{leaf}"
            k_u_w = f"{base}.u.weight"
            k_v_w = f"{base}.v.weight"
            k_u_b = f"{base}.u.bias"

            if (k_u_w in sd) and (k_v_w in sd):
                old = _get_module_by_dotted(model, f"decoder.{i}.{leaf}")
                assert isinstance(old, nn.Linear), f"Expected nn.Linear at decoder.{i}.{leaf}, got {type(old)}"

                in_f, out_f = old.in_features, old.out_features
                r_max = sd[k_v_w].shape[0]
                has_bias = (k_u_b in sd)

                tfl = SlicableTwoFactorLinear(in_features=in_f, out_features=out_f, r_max=r_max, bias=has_bias)
                tfl = tfl.to(device=old.weight.device, dtype=old.weight.dtype)
                tfl.V.copy_(sd[k_v_w].to(tfl.V.dtype))
                tfl.U.copy_(sd[k_u_w].to(tfl.U.dtype))
                if has_bias:
                    tfl.bias.copy_(sd[k_u_b].to(tfl.bias.dtype))
                _set_module_by_dotted(model, f"decoder.{i}.{leaf}", tfl)
    return model


def strip_factor_keys(sd: dict):
    sd2 = dict(sd)
    for k in list(sd2.keys()):
        if k.endswith(".u.weight") or k.endswith(".v.weight") or k.endswith(".u.bias"):
            sd2.pop(k, None)
    return sd2

def set_model_rank_frac(model: nn.Module, frac: float):
    for m in model.modules():
        if isinstance(m, SlicableTwoFactorLinear):
            m.set_active_frac(frac)



######################################################################################################################
######################################################################################################################
######################################################################################################################



def load_images(filelist: List[str], PIXEL_LIMIT: int = 255000, new_width: Optional[int] = None, verbose: bool = False):
    """
    Loads images from a directory or video, resizes them to a uniform size,
    then converts and stacks them into a single [N, 3, H, W] PyTorch tensor.
    """
    sources = [] 
    
    # --- 1. Load image paths or video frames ---
    for img_path in filelist:
        try:
            sources.append(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"Could not load image {img_path}: {e}")

    if not sources:
        print("No images found or loaded.")
        return torch.empty(0)

    if verbose:
        print(f"Found {len(sources)} images/frames. Processing...")

    # --- 2. Determine a uniform target size for all images based on the first image ---
    # This is necessary to ensure all tensors have the same dimensions for stacking.
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    if new_width is None:
        scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target, H_target = W_orig * scale, H_orig * scale
        k, m = round(W_target / 14), round(H_target / 14)
        while (k * 14) * (m * 14) > PIXEL_LIMIT:
            if k / m > W_target / H_target: k -= 1
            else: m -= 1
        TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    else:
        TARGET_W, TARGET_H = new_width, round(H_orig * (new_width / W_orig) / 14) * 14
    if verbose:
        print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    # --- 3. Resize images and convert them to tensors in the [0, 1] range ---
    tensor_list = []
    # Define a transform to convert a PIL Image to a CxHxW tensor and normalize to [0,1]
    to_tensor_transform = tvf.ToTensor()
    
    for img_pil in sources:
        try:
            # Resize to the uniform target size
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            # Convert to tensor
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    # --- 4. Stack the list of tensors into a single [N, C, H, W] batch tensor ---
    return torch.stack(tensor_list, dim=0)


def load_and_resize14(filelist: List[str], new_width: int, device: str, verbose: bool):
    imgs = load_images(filelist, new_width=new_width, verbose=verbose).to(device)

    ori_h, ori_w = imgs.shape[-2:]
    patch_h, patch_w = ori_h // 14, ori_w // 14
    # (N, 3, h, w) -> (1, N, 3, h_14, w_14)
    imgs = F.interpolate(imgs, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=False, antialias=True).unsqueeze(0)
    return imgs

def infer_monodepth(file: str, model: Pi3, hydra_cfg: DictConfig):

    imgs = load_and_resize14([file], new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    points = pred['local_points'][0]         # (1, h_14, w_14, 3)
    depth_map = points[0, ..., -1].detach()  # (h_14, w_14)
    return depth_map  # torch.Tensor



def adaptive_infer_monodepth(file: str, model: Pi3, hydra_cfg: DictConfig):
    """
        Adaptive inference for monodepth estimation.
    """

    imgs = load_and_resize14([file], new_width=hydra_cfg.load_img_size,
                             device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    # compute entropy score + map to retention
    # TODO

    # rr in {0.1, 0.2, 0.3} (these are "absolute retention ratios" wrt original full-rank)
    rr = 0.2  # TODO: from entropy mapping
    base_rr = 0.4

    # slice fraction relative to base checkpoint rank
    frac = min(1.0, rr / base_rr)
    set_model_rank_frac(model, frac)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    points = pred['local_points'][0]
    depth_map = points[0, ..., -1].detach()
    return depth_map

def infer_videodepth(filelist: str, model: Pi3, hydra_cfg: DictConfig):

    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    start = time.time()
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)
    end = time.time()

    depth_map = pred['local_points'][0, ..., -1]  # (N, h_14, w_14)
    depth_conf = pred['conf'][0, ..., 0]          # (N, h_14, w_14)
    return end - start, depth_map, depth_conf


def infer_cameras_w2c(filelist: str, model: Pi3, hydra_cfg: DictConfig):

    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    poses_c2w_all = pred['camera_poses'].cpu()
    extrinsics = se3_inverse(poses_c2w_all[0])

    return extrinsics, None


def infer_cameras_c2w(filelist: str, model: Pi3, hydra_cfg: DictConfig):

    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    poses_c2w_all = pred['camera_poses'].cpu()

    return poses_c2w_all[0], None

def infer_mv_pointclouds(filelist: str, model: Pi3, hydra_cfg: DictConfig, data_size: Tuple[int, int]):

    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)
    
    global_points = pred['points'][0]  # (N, h, w, 3)
    global_points = F.interpolate(
        global_points.permute(0, 3, 1, 2), data_size,
        mode="bilinear", align_corners=False, antialias=True
    ).permute(0, 2, 3, 1)  # align to gt

    return global_points.cpu().numpy()