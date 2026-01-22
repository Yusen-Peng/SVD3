import math
import torch
import torch.nn.functional as F
import torchvision.transforms as tvf
import time
import torch.nn as nn
from typing import List, Optional, Tuple
from omegaconf import DictConfig
from PIL import Image
from typing import Optional
import json
from typing import Dict, Any
from functools import lru_cache
import json
import torch
from typing import List, Dict, Any
from tqdm import tqdm
import cv2
import numpy as np

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pi3.models.pi3 import Pi3
from pi3.utils.geometry import se3_inverse

BASE_RR = 0.4

@lru_cache(maxsize=1)
def _load_entropy_cfg(path: str):
    with open(path, "r") as f:
        return json.load(f)

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
    
    def set_active_frac(self, frac: float):
        """
            Set the active ran fraction for this layer to support dynamic slicing.
        """
        self.active_frac = None if frac is None else float(frac)

    def forward(self, x, r: Optional[int] = None):
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

#################################################################################################################


"""
    Below is the implementation of entropy-guided data-adaptive inference.
"""

@torch.no_grad()
def entropy_score_from_imgs(imgs: torch.Tensor, bins: int = 256) -> float:
    assert imgs.ndim == 5 and imgs.shape[0] == 1 and imgs.shape[1] == 1 and imgs.shape[2] == 3, \
        f"Expected (1,1,3,H,W), got {tuple(imgs.shape)}"

    x = imgs[0, 0]  # (3,H,W)
    # grayscale
    gray = 0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2]  # (H,W)
    gray = gray.clamp(0, 1)

    # quantize to [0, bins-1]
    q = (gray * (bins - 1)).to(torch.int64)  # (H,W)

    hist = torch.bincount(q.flatten(), minlength=bins).float()
    p = hist / (hist.sum() + 1e-12)
    H = -(p * (p + 1e-12).log2()).sum()
    return float(H.item())

@torch.no_grad()
def normalize_entropy_score(s: float, cfg) -> float:
    """
    Percentile-based normalization using calibration statistics.

    s_norm = clip((s - p5) / (p95 - p5), 0, 1)

    Assumes entropy_p5 and entropy_p95 are computed on a calibration set.
    """
    p5  = float(cfg['entropy_p5'])
    p95 = float(cfg['entropy_p95'])

    denom = max(p95 - p5, 1e-6)
    s_norm = (s - p5) / denom
    return float(min(1.0, max(0.0, s_norm)))


@torch.no_grad()
def augmented_entropy_score_from_imgs(imgs: torch.Tensor, bins: int = 256) -> float:
    assert imgs.ndim == 5 and imgs.shape[0] == 1 and imgs.shape[1] == 1 and imgs.shape[2] == 3, \
        f"Expected (1,1,3,H,W), got {tuple(imgs.shape)}"

    x = imgs[0, 0]  # (3,H,W)
    # grayscale
    gray = 0.2989 * x[0] + 0.5870 * x[1] + 0.1140 * x[2]  # (H,W)
    gray = gray.clamp(0, 1)

    # part 1: entropy of the raw gray image
    # quantize to [0, bins-1]
    q = (gray * (bins - 1)).to(torch.int64)  # (H,W)

    hist = torch.bincount(q.flatten(), minlength=bins).float()
    p = hist / (hist.sum() + 1e-12)
    H_img = -(p * (p + 1e-12).log2()).sum()

    # part 2: entropy of binarization map
    gray_u8 = gray.mul(255.0).round().to(torch.uint8).cpu().numpy()
    _, bin_u8 = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bin_mask = (bin_u8 > 0).astype(np.uint8)  # {0,1}
    p1 = bin_mask.mean()          # foreground ratio
    p0 = 1.0 - p1
    eps = 1e-12
    H_bin = 0.0
    if p0 > 0:
        H_bin -= p0 * np.log2(p0 + eps)
    if p1 > 0:
        H_bin -= p1 * np.log2(p1 + eps)

    # part 3: entropy of Canny edges
    gray_blur = cv2.GaussianBlur(gray_u8, (5, 5), 0)
    edges_u8 = cv2.Canny(
        gray_blur,
        threshold1=50,
        threshold2=150,
        apertureSize=3,
        L2gradient=True
    )
    edge_mask = (edges_u8 > 0).astype(np.uint8)  # {0,1}
    p1 = edge_mask.mean()          # edge density in [0,1]
    p0 = 1.0 - p1
    eps = 1e-12
    H_edge = 0.0
    if p0 > 0:
        H_edge -= p0 * np.log2(p0 + eps)
    if p1 > 0:
        H_edge -= p1 * np.log2(p1 + eps)

    return H_img, H_bin, H_edge

@torch.no_grad()
def rr_from_entropy(s_norm: float, cfg: dict) -> float:
    th = cfg["rr_thresholds"]
    rr = [0.1, 0.2, 0.3] # 10%, 20%, 30% compression ratios

    if s_norm < th[0]:
        return rr[0]
    elif s_norm < th[1]:
        return rr[1]
    else:
        return rr[2]
    
@torch.no_grad()
def learn_entropy_cfg_from_calib(
    calib: List[Dict[str, torch.Tensor]],
    save_path: str = '/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg.json',
    bins: int = 256,
    tail_frac: float = 0.25, # 25% low, 50% mid, 25% high => avg rr = 0.2
    rr_values: Tuple[float, float, float] = (0.1, 0.2, 0.3),
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Build entropy cfg from calibration dataset, reusing user's helpers.

    Output JSON schema:
    {
      "entropy_p5": ...,
      "entropy_p95": ...,
      "rr_thresholds": [t0, t1],   # in normalized space [0,1]
      "rr_values": [0.1, 0.2, 0.3],
      "tail_frac": 0.25
    }

    Avg-retention control:
    - if rr_values = (0.1, 0.2, 0.3), symmetric tails guarantee E[rr] = 0.2
      under the calibration distribution.
    """
    assert 0.0 < tail_frac < 0.5, "tail_frac must be in (0, 0.5)"
    rr0, rr1, rr2 = rr_values
    assert abs(rr1 - 0.2) < 1e-9, "This avg-control trick assumes middle rr is 0.2."

    entropies = []

    for batch in calib:
        pv = batch["pixel_values"].to(device)  # (B,3,H,W)
        B = pv.shape[0]
        for b in range(B):
            imgs = pv[b].unsqueeze(0).unsqueeze(0)  # (1,1,3,H,W)
            H = entropy_score_from_imgs(imgs, bins=bins)
            entropies.append(H)

    if len(entropies) == 0:
        raise ValueError("Calibration dataset is empty; cannot learn entropy cfg.")

    ent = torch.tensor(entropies, dtype=torch.float32)

    entropy_p5 = float(torch.quantile(ent, 0.05).item())
    entropy_p95 = float(torch.quantile(ent, 0.95).item())

    # build cfg with just normalization stats first (so we can normalize)
    cfg: Dict[str, Any] = {
        "entropy_p5": entropy_p5,
        "entropy_p95": entropy_p95,
        "rr_values": [float(rr0), float(rr1), float(rr2)],
        "tail_frac": float(tail_frac),
    }

    s_norm_list = [normalize_entropy_score(float(s), cfg) for s in entropies]
    s_norm = torch.tensor(s_norm_list, dtype=torch.float32).clamp(0, 1)


    t0 = float(torch.quantile(s_norm, tail_frac).item())
    t1 = float(torch.quantile(s_norm, 1.0 - tail_frac).item())

    cfg["rr_thresholds"] = [t0, t1]

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(cfg, f, indent=2)
    print(f"🍀🍀🍀Saved adaptive entropy cfg to {save_path} 🍀🍀🍀")

    return cfg


#### fine-grained budget mapping ####
def sigmoid(x):
    # supports float or torch.Tensor
    if isinstance(x, torch.Tensor):
        return 1.0 / (1.0 + torch.exp(-x))
    else:
        return 1.0 / (1.0 + math.exp(-float(x)))


@torch.no_grad()
def rr_from_snorm_fine_grained(
    s_norm: torch.Tensor,  # shape: () or (N,)
    rr_min: float,
    rr_max: float,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    return rr_min + (rr_max - rr_min) * sigmoid(alpha * (s_norm - beta))


@torch.no_grad()
def solve_beta_for_budget(
    s_norm: torch.Tensor,          # (N,) in [0,1]
    rr_min: float = 0.1,
    rr_max: float = 0.3,
    rr_target: float = 0.2,
    alpha: float = 10.0,
    beta_low: float = -1.0,
    beta_high: float = 2.0,
    iters: int = 30, # 30 iterations
) -> float:
    """
        Binary search beta so that mean rr equals rr_target on calibration s_norm.
        Monotonicity: mean_rr(beta) decreases as beta increases.
    """
    assert s_norm.ndim == 1 and s_norm.numel() > 0
    s_norm = s_norm.clamp(0, 1)
    lo, hi = float(beta_low), float(beta_high)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        mean_rr = rr_from_snorm_fine_grained(s_norm, rr_min, rr_max, alpha, mid).mean().item()
        if mean_rr > rr_target:
            # too much retention
            # increase beta (shift right) to reduce rr
            lo = mid
        else:
            # too little retention
            # decrease beta (shift left) to increase rr
            hi = mid

    return 0.5 * (lo + hi)

@torch.no_grad()
def learn_entropy_cfg_continuous_from_calib(
    calib: List[Dict[str, torch.Tensor]],
    save_path: str = "/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_finegrained.json",
    bins: int = 256,
    rr_min: float = 0.1,
    rr_max: float = 0.3,
    rr_target: float = 0.2,
    alpha: float = 10.0,
    device: str = "cuda",
) -> Dict[str, Any]:

    entropies: List[float] = []

    for batch in calib:
        pv = batch["pixel_values"].to(device)  # (B,3,H,W)
        B = pv.shape[0]
        for b in range(B):
            imgs = pv[b].unsqueeze(0).unsqueeze(0)  # (1,1,3,H,W)
            H = entropy_score_from_imgs(imgs, bins=bins)
            entropies.append(float(H))

    if len(entropies) == 0:
        raise ValueError("Calibration dataset is empty; cannot learn entropy cfg.")

    ent = torch.tensor(entropies, dtype=torch.float32)

    entropy_p5 = float(torch.quantile(ent, 0.05).item())
    entropy_p95 = float(torch.quantile(ent, 0.95).item())

    # normalize to s_norm in [0,1]
    denom = max(entropy_p95 - entropy_p5, 1e-6)
    s_norm = ((ent - entropy_p5) / denom).clamp(0, 1)

    beta = solve_beta_for_budget(
        s_norm=s_norm,
        rr_min=rr_min,
        rr_max=rr_max,
        rr_target=rr_target,
        alpha=alpha,
        beta_low=-1.0,
        beta_high=2.0,
        iters=30,
    )

    cfg: Dict[str, Any] = {
        "entropy_p5": entropy_p5,
        "entropy_p95": entropy_p95,
        "bins": int(bins),
        "rr_min": float(rr_min),
        "rr_max": float(rr_max),
        "rr_target": float(rr_target),
        "alpha": float(alpha),
        "beta": float(beta),
        "mapping": "sigmoid_budget_continuous",
    }

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(cfg, f, indent=2)

    print(f"🍀🍀🍀Saved 🎃🎃Fine-Grained🎃🎃 adaptive entropy cfg to {save_path} 🍀🍀🍀")
    print(f"   alpha={alpha:.2f}, beta={beta:.4f}")
    return cfg




@torch.no_grad()
def rr_from_entropy_fine_grained_inference(s: float, cfg: dict) -> float:
    """
    Continuous mapping:
        rr = rr_min + (rr_max-rr_min) * sigmoid(alpha * (s_norm - beta))
    where:
        s_norm = clip((s - p5)/(p95-p5), 0, 1)
    """
    p5  = float(cfg["entropy_p5"])
    p95 = float(cfg["entropy_p95"])
    denom = max(p95 - p5, 1e-6)
    s_norm = (float(s) - p5) / denom
    s_norm = float(min(1.0, max(0.0, s_norm)))

    rr_min = float(cfg.get("rr_min", 0.1))
    rr_max = float(cfg.get("rr_max", 0.3))
    alpha  = float(cfg["alpha"])
    beta   = float(cfg["beta"])

    return rr_from_snorm_fine_grained(s_norm, rr_min, rr_max, alpha, beta)

########################### ablation #################################

def mix_max_norm(v, p5, p95):
    denom = max(p95 - p5, 1e-6)
    return (v - p5) / denom

@torch.no_grad()
def learn_augmented_entropy_cfg_from_calib(
    calib: List[Dict[str, torch.Tensor]],
    save_path: str = '/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_augmented.json',
    bins: int = 256,
    tail_frac: float = 0.25, # 25% low, 50% mid, 25% high => avg rr = 0.2
    rr_values: Tuple[float, float, float] = (0.1, 0.2, 0.3),
    device: str = "cuda",
) -> Dict[str, Any]:

    assert 0.0 < tail_frac < 0.5, "tail_frac must be in (0, 0.5)"
    rr0, rr1, rr2 = rr_values
    assert abs(rr1 - 0.2) < 1e-9, "This avg-control trick assumes middle rr is 0.2."


    # collect per-component entropies
    H_img_list  = []
    H_bin_list  = []
    H_edge_list = []


    for batch in calib:
        pv = batch["pixel_values"].to(device)  # (B,3,H,W)
        B = pv.shape[0]
        for b in range(B):
            imgs = pv[b].unsqueeze(0).unsqueeze(0)  # (1,1,3,H,W)
            H_img, H_bin, H_edge = augmented_entropy_score_from_imgs(imgs, bins=bins)

            H_img_list.append(H_img)
            H_bin_list.append(H_bin)
            H_edge_list.append(H_edge)

    if len(H_img_list) == 0 or len(H_bin_list) == 0 or len(H_edge_list) == 0:
        raise ValueError("Calibration dataset is empty; cannot learn entropy cfg.")

    H_img_t  = torch.tensor(H_img_list, dtype=torch.float32)
    H_bin_t  = torch.tensor(H_bin_list, dtype=torch.float32)
    H_edge_t = torch.tensor(H_edge_list, dtype=torch.float32)


    # per-component p5 / p95
    img_p5  = float(torch.quantile(H_img_t,  0.05).item())
    img_p95 = float(torch.quantile(H_img_t,  0.95).item())
    bin_p5  = float(torch.quantile(H_bin_t,  0.05).item())
    bin_p95 = float(torch.quantile(H_bin_t,  0.95).item())
    edge_p5  = float(torch.quantile(H_edge_t, 0.05).item())
    edge_p95 = float(torch.quantile(H_edge_t, 0.95).item())

    # build summed normalized score
    s_norm_list = []
    for h_img, h_bin, h_edge in zip(H_img_list, H_bin_list, H_edge_list):
        h_img_n  = mix_max_norm(h_img,  img_p5,  img_p95)
        h_bin_n  = mix_max_norm(h_bin,  bin_p5,  bin_p95)
        h_edge_n = mix_max_norm(h_edge, edge_p5, edge_p95)

        s = h_img_n + h_bin_n + h_edge_n # a simple sum of **normalized** scores
        s_norm_list.append(s)

    s_norm = torch.tensor(s_norm_list, dtype=torch.float32).clamp(0, 1e9)

    # thresholds in summed-score space
    t0 = float(torch.quantile(s_norm, tail_frac).item())
    t1 = float(torch.quantile(s_norm, 1.0 - tail_frac).item())

    # final cfg
    cfg: Dict[str, Any] = {
        "img_entropy_p5": img_p5,
        "img_entropy_p95": img_p95,
        "bin_entropy_p5": bin_p5,
        "bin_entropy_p95": bin_p95,
        "edge_entropy_p5": edge_p5,
        "edge_entropy_p95": edge_p95,
        "rr_values": [float(rr0), float(rr1), float(rr2)],
        "tail_frac": float(tail_frac),
    }

    cfg["rr_thresholds"] = [t0, t1]

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(cfg, f, indent=2)

    print(f"🍀🍀🍀Saved 😈😈augmented😈😈 entropy cfg to {save_path} 🍀🍀🍀")
    return cfg


def adaptive_infer_monodepth(file: str, model: Pi3, hydra_cfg: DictConfig):
    """
        Adaptive inference for monodepth estimation.
    """

    imgs = load_and_resize14([file], new_width=hydra_cfg.load_img_size,
                             device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    # compute entropy score + map to retention
    entropy_cfg = _load_entropy_cfg('/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg.json')
    s = entropy_score_from_imgs(imgs, bins=256)
    s_norm = normalize_entropy_score(s, entropy_cfg)
    rr = rr_from_entropy(s_norm, entropy_cfg)

    # slice fraction relative to base checkpoint rank
    frac = min(1.0, rr / BASE_RR)
    set_model_rank_frac(model, frac)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    points = pred['local_points'][0]
    depth_map = points[0, ..., -1].detach()
    return depth_map


@torch.no_grad()
def fine_grained_adaptive_infer_monodepth(file: str, model: Pi3, hydra_cfg: DictConfig):
    """
    Fine-grained (continuous) adaptive inference for monodepth estimation.
    Uses learned (p5,p95,beta) + fixed alpha to produce a continuous rr(x).
    """

    imgs = load_and_resize14([file], new_width=hydra_cfg.load_img_size,
                            device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    # compute entropy score + map to continuous retention
    entropy_cfg = _load_entropy_cfg("/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_finegrained.json")
    s = entropy_score_from_imgs(imgs, bins=int(entropy_cfg.get("bins", 256)))
    rr = rr_from_entropy_fine_grained_inference(s, entropy_cfg)

    # slice fraction relative to base checkpoint rank
    frac = min(1.0, rr / BASE_RR)
    set_model_rank_frac(model, frac)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
        pred = model(imgs)

    points = pred["local_points"][0]
    depth_map = points[0, ..., -1].detach()
    return depth_map



def augmented_adaptive_infer_monodepth(file: str, model: Pi3, hydra_cfg: DictConfig):
    """
        Adaptive inference for monodepth estimation using augmented entropy.
    """

    imgs = load_and_resize14([file], new_width=hydra_cfg.load_img_size,
                             device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    # compute augmented entropy score + map to retention
    entropy_cfg = _load_entropy_cfg('/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_augmented.json')
    H_img, H_bin, H_edge = augmented_entropy_score_from_imgs(imgs, bins=256)

    h_img_n  = mix_max_norm(H_img,  entropy_cfg["img_entropy_p5"],  entropy_cfg["img_entropy_p95"])
    h_bin_n  = mix_max_norm(H_bin,  entropy_cfg["bin_entropy_p5"],  entropy_cfg["bin_entropy_p95"])
    h_edge_n = mix_max_norm(H_edge, entropy_cfg["edge_entropy_p5"], entropy_cfg["edge_entropy_p95"])

    s = h_img_n + h_bin_n + h_edge_n # a simple sum of **normalized** scores
    # s = h_bin_n + h_edge_n # FIXME: ablation without image entropy
    s_norm = float(max(0.0, s))
    rr = rr_from_entropy(s_norm, entropy_cfg)

    # slice fraction relative to base checkpoint rank
    frac = min(1.0, rr / BASE_RR)
    set_model_rank_frac(model, frac)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    points = pred['local_points'][0]
    depth_map = points[0, ..., -1].detach()
    return depth_map

############################################################################################################


"""
    Below is the implementation of embedding-entropy-guided data-adaptive inference.
"""

@torch.no_grad()
def entropy_score_from_embeddings(
    patch_tokens: torch.Tensor,
    codebook: torch.Tensor,
    tau: float = 10.0,
    eps: float = 1e-8,
    soft: bool = True,
) -> float:
    """
    patch_tokens: (L, D) encoder patch embeddings for ONE image
    codebook:     (K, D) prototype vectors
    Returns: normalized entropy in [0,1] (approximately)
    """
    X = patch_tokens.float()
    C = codebook.float()

    # cosine k-means-ish
    X = F.normalize(X, dim=-1)
    C = F.normalize(C, dim=-1)

    logits = tau * (X @ C.t())  # (L, K)

    if soft:
        A = logits.softmax(dim=-1)
        H = -(A * (A + eps).log()).sum(dim=-1).mean()
        Hn = H / math.log(A.shape[-1] + 1e-12)
        return float(Hn.item())
    else:
        k = logits.argmax(dim=-1)         # (L,)
        p = torch.bincount(k, minlength=C.shape[0]).float()
        p = p / p.sum().clamp_min(eps)

    H = -(p * (p + eps).log()).sum()
    Hn = H / math.log(p.numel() + 1e-12)
    return float(Hn.item())

@torch.no_grad()
def build_codebook_kmeans(
    all_tokens: torch.Tensor,   # (M, D)
    K: int,
    iters: int = 10,
    seed: int = 0,
) -> torch.Tensor:
    """
    Simple cosine k-means on GPU.
    Returns: (K, D) centroids
    """
    device = all_tokens.device
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    X = F.normalize(all_tokens.float(), dim=-1)

    # init: random samples
    idx = torch.randperm(X.shape[0], generator=g, device=device)[:K]
    C = X[idx].clone()  # (K, D)

    for _ in range(iters):
        # assign
        sim = X @ C.T              # (M, K)
        labels = sim.argmax(dim=1)

        # update
        for k in range(K):
            mask = labels == k
            if mask.any():
                C[k] = X[mask].mean(dim=0)
        C = F.normalize(C, dim=-1)

    return C


@torch.no_grad()
def learn_entropy_cfg_from_calib_embedding(
    calib: List[Dict[str, torch.Tensor]],
    model: Pi3,
    save_path: str = "/mnt/extdisk1/wanghaoxuan/SVD-pi3/embedding_adaptive_cfg.json",
    tail_frac: float = 0.25,
    rr_values: Tuple[float, float, float] = (0.1, 0.2, 0.3),
    device: str = "cuda",
    K: int = 256,                # codebook size
    tau: float = 10.0,           # soft assignment temperature
    soft: bool = True,           # soft vs hard assignment
    max_tokens_per_image: int = 256,  # subsample L tokens to speed up entropy
    max_total_tokens: int = 200_000,  # cap total tokens for codebook building
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Learns an entropy cfg but using encoder patch embeddings instead of pixels.

    Output JSON schema (extends yours by adding codebook + params):
    {
      "entropy_p5": ...,
      "entropy_p95": ...,
      "rr_thresholds": [t0, t1],
      "rr_values": [...],
      "tail_frac": ...,

      "embed_entropy": {
        "kind": "codebook_softassign",
        "K": 256,
        "tau": 10.0,
        "soft": true,
        "max_tokens_per_image": 256,
        "seed": 0
      }
    }
    """
    assert 0.0 < tail_frac < 0.5, "tail_frac must be in (0, 0.5)"
    rr0, rr1, rr2 = rr_values
    assert abs(rr1 - 0.2) < 1e-9, "This avg-control trick assumes middle rr is 0.2."

    model = model.to(device).eval()

    # collect patch tokens from calib to build a codebook
    token_bank = []

    for batch in calib:
        pv = batch["pixel_values"].to(device)  # (B,3,H,W)
        B = pv.shape[0]
        for b in range(B):
            img = pv[b].unsqueeze(0)  # (1,3,H,W)
            img = (img - model.image_mean) / model.image_std

            out = model.encoder(img, is_training=True)
            if isinstance(out, dict):
                out = out["x_norm_patchtokens"]          # (1, L, D)
            tokens: torch.Tensor = out[0]                               # (L, D)

            # subsample tokens per image for speed
            L = tokens.shape[0]
            if max_tokens_per_image is not None and L > max_tokens_per_image:
                idx = torch.randperm(L, device=device)[:max_tokens_per_image]
                tokens = tokens[idx]

            token_bank.append(tokens.detach())

    if len(token_bank) == 0:
        raise ValueError("Calibration dataset is empty; cannot learn embedding entropy cfg.")

    all_tokens = torch.cat(token_bank, dim=0)  # (M, D)

    # cap total tokens for memory / speed
    if max_total_tokens is not None and all_tokens.shape[0] > max_total_tokens:
        idx = torch.randperm(all_tokens.shape[0], device=device)[:max_total_tokens]
        all_tokens = all_tokens[idx]

    # build codebook (using k-means)
    codebook = build_codebook_kmeans(all_tokens, K=K, iters=10)

    # compute per-image embedding entropy scores
    entropies = []
    for tokens in token_bank:
        H = entropy_score_from_embeddings(tokens, codebook, tau=tau, soft=soft)
        entropies.append(H)

    ent = torch.tensor(entropies, dtype=torch.float32)

    entropy_p5 = float(torch.quantile(ent, 0.05).item())
    entropy_p95 = float(torch.quantile(ent, 0.95).item())

    # build cfg with normalization stats
    cfg: Dict[str, Any] = {
        "entropy_p5": entropy_p5,
        "entropy_p95": entropy_p95,
        "rr_values": [float(rr0), float(rr1), float(rr2)],
        "tail_frac": float(tail_frac),
        "embed_entropy": {
            "kind": "codebook_softassign" if soft else "codebook_hardassign",
            "K": int(K),
            "tau": float(tau),
            "soft": bool(soft),
            "max_tokens_per_image": int(max_tokens_per_image) if max_tokens_per_image is not None else None,
            "seed": int(seed),
        },
        "codebook": codebook.detach().float().cpu().tolist(),
    }

    # thresholds in normalized space (reuse your normalize_entropy_score helper)
    s_norm_list = [normalize_entropy_score(float(s), cfg) for s in entropies]
    s_norm = torch.tensor(s_norm_list, dtype=torch.float32).clamp(0, 1)

    t0 = float(torch.quantile(s_norm, tail_frac).item())
    t1 = float(torch.quantile(s_norm, 1.0 - tail_frac).item())
    cfg["rr_thresholds"] = [t0, t1]

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(cfg, f, indent=2)
    print(f"🩵🩵🩵Saved adaptive EMBED entropy cfg to {save_path}🩵🩵🩵")

    return cfg

@torch.no_grad()
def entropy_score_from_imgs_embedding(
    model: Pi3,
    imgs: torch.Tensor,          # (B,N,3,H,W)
    entropy_cfg: dict,
    device: str = "cuda",
    eps: float = 1e-8,
) -> float:
    """
    Compute embedding-based entropy using encoder patch tokens + codebook.
    Uses the SAME cfg normalization helpers as pixel entropy.
    """
    # ---- load codebook + params ----
    codebook = torch.tensor(entropy_cfg["codebook"], device=device, dtype=torch.float32)  # (K,D)
    tau = float(entropy_cfg.get("embed_entropy", {}).get("tau", 10.0))
    soft = bool(entropy_cfg.get("embed_entropy", {}).get("soft", True))
    max_tokens = entropy_cfg.get("embed_entropy", {}).get("max_tokens_per_image", None)

    B, N, C, H, W = imgs.shape
    imgs_bn = imgs.reshape(B * N, C, H, W).to(device)

    # ---- match Pi3.forward normalization for encoder ----
    imgs_bn = (imgs_bn - model.image_mean.to(device)) / model.image_std.to(device)

    # ---- encoder only ----
    out = model.encoder(imgs_bn, is_training=True)
    if isinstance(out, dict):
        out = out["x_norm_patchtokens"]   # (BN, L, D)

    # ---- entropy per frame, then average ----
    BN, L, D = out.shape
    Cb = F.normalize(codebook, dim=-1)

    ent_list = []
    for i in range(BN):
        X = out[i].float()                # (L, D)
        X = F.normalize(X, dim=-1)

        if max_tokens is not None and L > int(max_tokens):
            idx = torch.randperm(L, device=device)[: int(max_tokens)]
            X = X[idx]

        logits = tau * (X @ Cb.t())       # (L, K)
        if soft:
            A = logits.softmax(dim=-1)
            H = -(A * (A + eps).log()).sum(dim=-1).mean()
            Hn = H / math.log(A.shape[-1] + 1e-12)
            return float(Hn.item())
        else:
            k = logits.argmax(dim=-1)
            p = torch.bincount(k, minlength=Cb.shape[0]).float()
            p = p / p.sum().clamp_min(eps)

        H = -(p * (p + eps).log()).sum()
        Hn = H / torch.log(torch.tensor(float(p.numel()), device=device))
        ent_list.append(Hn)

    return float(torch.stack(ent_list).mean().item())


def embedding_adaptive_infer_monodepth(file: str, model: Pi3, hydra_cfg: DictConfig):
    """
    Adaptive inference for monodepth using encoder-embedding entropy (codebook soft assignment).
    """
    imgs = load_and_resize14(
        [file],
        new_width=hydra_cfg.load_img_size,
        device=hydra_cfg.device,
        verbose=hydra_cfg.verbose
    )

    # compute EMBEDDING entropy score + map to retention
    entropy_cfg = _load_entropy_cfg('/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_embedding.json')
    s = entropy_score_from_imgs_embedding(model, imgs, entropy_cfg, device=str(hydra_cfg.device))
    s_norm = normalize_entropy_score(s, entropy_cfg)
    rr = rr_from_entropy(s_norm, entropy_cfg)

    # slice fraction relative to base checkpoint rank
    frac = min(1.0, rr / BASE_RR)
    set_model_rank_frac(model, frac)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    points = pred['local_points'][0]
    depth_map = points[0, ..., -1].detach()
    return depth_map


############################################################################################################

"""
    Below is the implementation of cossim-drift-guided data-adaptive inference.
"""

@torch.no_grad()
def _cosine_drift_tokens(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x, y: (..., T, D)
    returns: scalar drift tensor
    """
    x_f = x.reshape(-1, x.shape[-1]).float()
    y_f = y.reshape(-1, y.shape[-1]).float()
    cos = F.cosine_similarity(x_f, y_f, dim=-1, eps=eps)
    return (1.0 - cos).mean()

@torch.no_grad()
def probe_cosine_drift_early_decoder(
    model: Pi3,
    imgs: torch.Tensor,          # (B,N,3,H,W) already resized like load_and_resize14 gives you
    probe_layers: int = 4,
    ignore_special_tokens: bool = True,
    device: str = "cuda",
    eps: float = 1e-8,
) -> float:
    """
    Computes cosine drift between input/output of early decoder blocks without modifying Pi3.

    Returns: float scalar (mean drift over first `probe_layers` blocks and all frames)
    """
    model = model.to(device).eval()
    imgs = imgs.to(device)
    imgs = (imgs - model.image_mean.to(device)) / model.image_std.to(device)
    B, N, C, H, W = imgs.shape
    assert C == 3, f"Expected RGB imgs, got C={C}"
    imgs_bn = imgs.reshape(B * N, C, H, W)
    hidden = model.encoder(imgs_bn, is_training=True)
    if isinstance(hidden, dict):
        hidden = hidden["x_norm_patchtokens"]  # (BN, L, D)
    # hidden: (BN, hw, D)
    BN, hw, D = hidden.shape
    assert BN == B * N
    hidden = hidden.reshape(B * N, hw, D)

    # register tokens
    reg = model.register_token.repeat(B, N, 1, 1).reshape(B * N, *model.register_token.shape[-2:])  # (BN, S, D)
    hidden = torch.cat([reg, hidden], dim=1)  # (BN, S+hw, D)
    T = hidden.shape[1]

    # positions
    if not (hasattr(model, "pos_type") and str(model.pos_type).startswith("rope")):
        raise RuntimeError("This static probe currently supports rope pos_type only (matches your Pi3).")
    pos = model.position_getter(B * N, H // model.patch_size, W // model.patch_size, hidden.device)  # (BN, hw, 2)

    # handle special tokens position padding exactly like decode()
    if model.patch_start_idx > 0:
        pos = pos + 1
        pos_special = torch.zeros(B * N, model.patch_start_idx, 2, device=hidden.device, dtype=pos.dtype)
        pos = torch.cat([pos_special, pos], dim=1)  # (BN, T, 2)

    # run first probe_layers blocks and compute drift
    L0 = min(int(probe_layers), len(model.decoder))
    drifts = []

    for i in range(L0):
        blk = model.decoder[i]

        if i % 2 == 0:
            # (BN, T, D)
            pos_i = pos.reshape(B * N, T, -1)
            hid_i = hidden.reshape(B * N, T, -1)
        else:
            # (B, N*T, D)
            pos_i = pos.reshape(B, N * T, -1)
            hid_i = hidden.reshape(B, N * T, -1)

        x_in = hid_i
        y_out = blk(hid_i, xpos=pos_i)

        # choose tokens to score
        if ignore_special_tokens and model.patch_start_idx > 0:
            x_use = x_in[..., model.patch_start_idx:, :]
            y_use = y_out[..., model.patch_start_idx:, :]
        else:
            x_use, y_use = x_in, y_out


        drifts.append(_cosine_drift_tokens(x_use, y_use, eps=eps))

        if i % 2 == 0:
            hidden = y_out.reshape(B * N, T, -1)
        else:
            hidden = y_out.reshape(B * N, T, -1)
    score = torch.stack(drifts).mean()
    return float(score.item())

def normalize_probe_score(s: float, cfg: dict, eps: float = 1e-8) -> float:
    x = (s - cfg["score_p5"]) / (cfg["score_p95"] - cfg["score_p5"] + eps)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


@torch.no_grad()
def learn_drift_cfg_from_calib(
    calib: List[Dict[str, torch.Tensor]],
    model: Pi3,
    save_path: str = "/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_drift.json",
    tail_frac: float = 0.25,
    rr_values: Tuple[float, float, float] = (0.1, 0.2, 0.3),
    device: str = "cuda",
    probe_layers: int = 4,
    ignore_special_tokens: bool = True,
) -> Dict[str, Any]:
    assert 0.0 < tail_frac < 0.5, "tail_frac must be in (0, 0.5)"
    rr0, rr1, rr2 = rr_values
    assert abs(rr1 - 0.2) < 1e-9, "This avg-control trick assumes middle rr is 0.2."

    model = model.to(device).eval()

    drift_scores: List[float] = []

    for batch in calib:
        pv = batch["pixel_values"].to(device)  # (B,3,H,W)
        B = pv.shape[0]
        for b in range(B):
            imgs = pv[b].unsqueeze(0).unsqueeze(0)  # (1,1,3,H,W)
            s = probe_cosine_drift_early_decoder(
                model=model,
                imgs=imgs,
                probe_layers=probe_layers,
                ignore_special_tokens=ignore_special_tokens,
                device=device,
            )
            drift_scores.append(float(s))

    if len(drift_scores) == 0:
        raise ValueError("Calibration dataset is empty; cannot learn drift cfg.")

    s = torch.tensor(drift_scores, dtype=torch.float32)

    score_p5 = float(torch.quantile(s, 0.05).item())
    score_p95 = float(torch.quantile(s, 0.95).item())

    cfg: Dict[str, Any] = {
        "score_p5": score_p5,
        "score_p95": score_p95,
        "rr_values": [float(rr0), float(rr1), float(rr2)],
        "tail_frac": float(tail_frac),
        "drift_probe": {
            "kind": "cosine_drift_early_decoder",
            "probe_layers": int(probe_layers),
            "ignore_special_tokens": bool(ignore_special_tokens),
        },
    }

    s_norm_list = [normalize_probe_score(float(v), cfg) for v in drift_scores]
    s_norm = torch.tensor(s_norm_list, dtype=torch.float32).clamp(0, 1)

    t0 = float(torch.quantile(s_norm, tail_frac).item())
    t1 = float(torch.quantile(s_norm, 1.0 - tail_frac).item())
    cfg["rr_thresholds"] = [t0, t1]

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(cfg, f, indent=2)
    print(f"✅ Saved adaptive DRIFT cfg to {save_path}")

    return cfg

def _load_drift_cfg(path: str):
    with open(path, "r") as f:
        return json.load(f)



def drifting_adaptive_infer_monodepth(file: str, model: Pi3, hydra_cfg: DictConfig):
    """
    Adaptive inference for monodepth using early layer cos-sim drifting.
    """
    imgs = load_and_resize14(
        [file],
        new_width=hydra_cfg.load_img_size,
        device=hydra_cfg.device,
        verbose=hydra_cfg.verbose
    )

    # compute drifting score + map to retention
    drift_cfg = _load_drift_cfg('/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_drifting.json')
    s = probe_cosine_drift_early_decoder(
        model=model,
        imgs=imgs,
        probe_layers=drift_cfg.get("drift_probe", {}).get("probe_layers", 4),
        ignore_special_tokens=drift_cfg.get("drift_probe", {}).get("ignore_special_tokens", True),
        device=str(hydra_cfg.device),
    )
    s_norm = normalize_probe_score(s, drift_cfg)
    rr = rr_from_entropy(s_norm, drift_cfg)

    # slice fraction relative to base checkpoint rank
    frac = min(1.0, rr / BASE_RR)
    set_model_rank_frac(model, frac)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    points = pred['local_points'][0]
    depth_map = points[0, ..., -1].detach()
    return depth_map


########################################################################################################
########################################################################################################


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


def adaptive_infer_videodepth(filelist: str, model: Pi3, hydra_cfg: DictConfig):

    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    # compute entropy score + map to retention
    entropy_cfg = _load_entropy_cfg('/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg.json')
    
    # first image/frame only for entropy computation
    first = imgs[:, :1]   # -> (B, 1, 3, H, W) = (1, 1, 3, H, W)

    s = entropy_score_from_imgs(first, bins=256)
    s_norm = normalize_entropy_score(s, entropy_cfg)
    rr = rr_from_entropy(s_norm, entropy_cfg)

    # slice fraction relative to base checkpoint rank
    frac = min(1.0, rr / BASE_RR)
    set_model_rank_frac(model, frac)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    start = time.time()
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)
    end = time.time()

    depth_map = pred['local_points'][0, ..., -1]  # (N, h_14, w_14)
    depth_conf = pred['conf'][0, ..., 0]          # (N, h_14, w_14)
    return end - start, depth_map, depth_conf


def fine_grained_adaptive_infer_videodepth(filelist: str, model: Pi3, hydra_cfg: DictConfig):
    
    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    # compute entropy score + map to continuous retention
    entropy_cfg = _load_entropy_cfg("/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_finegrained.json")
    
    # first image/frame only for entropy computation
    first = imgs[:, :1]   # -> (B, 1, 3, H, W) = (1, 1, 3, H, W)

    s = entropy_score_from_imgs(first, bins=int(entropy_cfg.get("bins", 256)))
    rr = rr_from_entropy_fine_grained_inference(s, entropy_cfg)

    # slice fraction relative to base checkpoint rank
    frac = min(1.0, rr / BASE_RR)
    set_model_rank_frac(model, frac)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    start = time.time()
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)
    end = time.time()

    depth_map = pred['local_points'][0, ..., -1]  # (N, h_14, w_14)
    depth_conf = pred['conf'][0, ..., 0]          # (N, h_14, w_14)
    return end - start, depth_map, depth_conf




def augmented_adaptive_infer_videodepth(filelist: str, model: Pi3, hydra_cfg: DictConfig):
    
    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    # compute augmented entropy score + map to retention
    entropy_cfg = _load_entropy_cfg('/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_augmented.json')
    
    # first image/frame only for entropy computation
    first = imgs[:, :1]   # -> (B, 1, 3, H, W) = (1, 1, 3, H, W)

    H_img, H_bin, H_edge = augmented_entropy_score_from_imgs(first, bins=256)

    h_img_n  = mix_max_norm(H_img,  entropy_cfg["img_entropy_p5"],  entropy_cfg["img_entropy_p95"])
    h_bin_n  = mix_max_norm(H_bin,  entropy_cfg["bin_entropy_p5"],  entropy_cfg["bin_entropy_p95"])
    h_edge_n = mix_max_norm(H_edge, entropy_cfg["edge_entropy_p5"], entropy_cfg["edge_entropy_p95"])

    s = h_img_n + h_bin_n + h_edge_n # a simple sum of **normalized** scores
    s_norm = float(max(0.0, s))
    rr = rr_from_entropy(s_norm, entropy_cfg)

    # slice fraction relative to base checkpoint rank
    frac = min(1.0, rr / BASE_RR)
    set_model_rank_frac(model, frac)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    start = time.time()
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)
    end = time.time()

    depth_map = pred['local_points'][0, ..., -1]  # (N, h_14, w_14)
    depth_conf = pred['conf'][0, ..., 0]          # (N, h_14, w_14)
    return end - start, depth_map, depth_conf



def drifting_adaptive_infer_videodepth(filelist: str, model: Pi3, hydra_cfg: DictConfig):
    """
    Adaptive inference for monodepth using early layer cos-sim drifting.
    """
    imgs = load_and_resize14(
        filelist,
        new_width=hydra_cfg.load_img_size,
        device=hydra_cfg.device,
        verbose=hydra_cfg.verbose
    )

    # compute drifting score + map to retention
    drift_cfg = _load_drift_cfg('/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_drifting.json')
    
    first = imgs[:, :1]   # -> (B, 1, 3, H, W) = (1, 1, 3, H, W)
    
    s = probe_cosine_drift_early_decoder(
        model=model,
        imgs=first,
        probe_layers=drift_cfg.get("drift_probe", {}).get("probe_layers", 4),
        ignore_special_tokens=drift_cfg.get("drift_probe", {}).get("ignore_special_tokens", True),
        device=str(hydra_cfg.device),
    )
    s_norm = normalize_probe_score(s, drift_cfg)
    rr = rr_from_entropy(s_norm, drift_cfg)

    # slice fraction relative to base checkpoint rank
    frac = min(1.0, rr / BASE_RR)
    set_model_rank_frac(model, frac)

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


def adaptive_infer_cameras_c2w(filelist: str, model: Pi3, hydra_cfg: DictConfig):

    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    # compute entropy score + map to retention
    entropy_cfg = _load_entropy_cfg('/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg.json')
    first = imgs[:, :1] # select first image only for entropy computation
    s = entropy_score_from_imgs(first, bins=256)
    s_norm = normalize_entropy_score(s, entropy_cfg)
    rr = rr_from_entropy(s_norm, entropy_cfg)

    # slice fraction relative to base checkpoint rank
    frac = min(1.0, rr / BASE_RR)
    set_model_rank_frac(model, frac)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    poses_c2w_all = pred['camera_poses'].cpu()

    return poses_c2w_all[0], None


def fine_grained_adaptive_infer_cameras_c2w(filelist: str, model: Pi3, hydra_cfg: DictConfig):
    
    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    # compute entropy score + map to continuous retention
    entropy_cfg = _load_entropy_cfg("/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_finegrained.json")
    
    # first image/frame only for entropy computation
    first = imgs[:, :1]   # -> (B, 1, 3, H, W) = (1, 1, 3, H, W)

    s = entropy_score_from_imgs(first, bins=int(entropy_cfg.get("bins", 256)))
    rr = rr_from_entropy_fine_grained_inference(s, entropy_cfg)

    # slice fraction relative to base checkpoint rank
    frac = min(1.0, rr / BASE_RR)
    set_model_rank_frac(model, frac)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    poses_c2w_all = pred['camera_poses'].cpu()

    return poses_c2w_all[0], None


def augmented_adaptive_infer_cameras_c2w(filelist: str, model: Pi3, hydra_cfg: DictConfig):
    
    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    # compute augmented entropy score + map to retention
    entropy_cfg = _load_entropy_cfg('/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_augmented.json')
    
    # first image/frame only for entropy computation
    first = imgs[:, :1]   # -> (B, 1, 3, H, W) = (1, 1, 3, H, W)

    H_img, H_bin, H_edge = augmented_entropy_score_from_imgs(first, bins=256)

    h_img_n  = mix_max_norm(H_img,  entropy_cfg["img_entropy_p5"],  entropy_cfg["img_entropy_p95"])
    h_bin_n  = mix_max_norm(H_bin,  entropy_cfg["bin_entropy_p5"],  entropy_cfg["bin_entropy_p95"])
    h_edge_n = mix_max_norm(H_edge, entropy_cfg["edge_entropy_p5"], entropy_cfg["edge_entropy_p95"])

    s = h_img_n + h_bin_n + h_edge_n # a simple sum of **normalized** scores
    s_norm = float(max(0.0, s))
    rr = rr_from_entropy(s_norm, entropy_cfg)

    # slice fraction relative to base checkpoint rank
    frac = min(1.0, rr / BASE_RR)
    set_model_rank_frac(model, frac)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            pred = model(imgs)

    poses_c2w_all = pred['camera_poses'].cpu()

    return poses_c2w_all[0], None




def drifting_adaptive_infer_cameras_c2w(filelist: str, model: Pi3, hydra_cfg: DictConfig):

    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    # compute drifting score + map to retention
    drift_cfg = _load_drift_cfg('/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_drifting.json')
    
    first = imgs[:, :1]   # -> (B, 1, 3, H, W) = (1, 1, 3, H, W)
    
    s = probe_cosine_drift_early_decoder(
        model=model,
        imgs=first,
        probe_layers=drift_cfg.get("drift_probe", {}).get("probe_layers", 4),
        ignore_special_tokens=drift_cfg.get("drift_probe", {}).get("ignore_special_tokens", True),
        device=str(hydra_cfg.device),
    )
    s_norm = normalize_probe_score(s, drift_cfg)
    rr = rr_from_entropy(s_norm, drift_cfg)

    # slice fraction relative to base checkpoint rank
    frac = min(1.0, rr / BASE_RR)
    set_model_rank_frac(model, frac)

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

def adaptive_infer_mv_pointclouds(filelist: str, model: Pi3, hydra_cfg: DictConfig, data_size: Tuple[int, int]):

    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    # compute entropy score + map to retention
    entropy_cfg = _load_entropy_cfg('/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg.json')
    first = imgs[:, :1] # select first image only for entropy computation
    s = entropy_score_from_imgs(first, bins=256)
    s_norm = normalize_entropy_score(s, entropy_cfg)
    rr = rr_from_entropy(s_norm, entropy_cfg)

    # slice fraction relative to base checkpoint rank
    frac = min(1.0, rr / BASE_RR)
    set_model_rank_frac(model, frac)

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



def fine_grained_adaptive_infer_mv_pointclouds(filelist: str, model: Pi3, hydra_cfg: DictConfig, data_size: Tuple[int, int]):
    
    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    # compute entropy score + map to continuous retention
    entropy_cfg = _load_entropy_cfg("/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_finegrained.json")
    
    # first image/frame only for entropy computation
    first = imgs[:, :1]   # -> (B, 1, 3, H, W) = (1, 1, 3, H, W)

    s = entropy_score_from_imgs(first, bins=int(entropy_cfg.get("bins", 256)))
    rr = rr_from_entropy_fine_grained_inference(s, entropy_cfg)

    # slice fraction relative to base checkpoint rank
    frac = min(1.0, rr / BASE_RR)
    set_model_rank_frac(model, frac)

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


def augmented_adaptive_infer_mv_pointclouds(filelist: str, model: Pi3, hydra_cfg: DictConfig, data_size: Tuple[int, int]):
    
    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    # compute augmented entropy score + map to retention
    entropy_cfg = _load_entropy_cfg('/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_augmented.json')
    
    # first image/frame only for entropy computation
    first = imgs[:, :1]   # -> (B, 1, 3, H, W) = (1, 1, 3, H, W)

    H_img, H_bin, H_edge = augmented_entropy_score_from_imgs(first, bins=256)

    h_img_n  = mix_max_norm(H_img,  entropy_cfg["img_entropy_p5"],  entropy_cfg["img_entropy_p95"])
    h_bin_n  = mix_max_norm(H_bin,  entropy_cfg["bin_entropy_p5"],  entropy_cfg["bin_entropy_p95"])
    h_edge_n = mix_max_norm(H_edge, entropy_cfg["edge_entropy_p5"], entropy_cfg["edge_entropy_p95"])

    s = h_img_n + h_bin_n + h_edge_n # a simple sum of **normalized** scores
    s_norm = float(max(0.0, s))
    rr = rr_from_entropy(s_norm, entropy_cfg)

    # slice fraction relative to base checkpoint rank
    frac = min(1.0, rr / BASE_RR)
    set_model_rank_frac(model, frac)

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






def drifting_adaptive_infer_mv_pointclouds(filelist: str, model: Pi3, hydra_cfg: DictConfig, data_size: Tuple[int, int]):

    imgs = load_and_resize14(filelist, new_width=hydra_cfg.load_img_size, device=hydra_cfg.device, verbose=hydra_cfg.verbose)

    # compute drifting score + map to retention
    drift_cfg = _load_drift_cfg('/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_drifting.json')
    
    first = imgs[:, :1]   # -> (B, 1, 3, H, W) = (1, 1, 3, H, W)
    
    s = probe_cosine_drift_early_decoder(
        model=model,
        imgs=first,
        probe_layers=drift_cfg.get("drift_probe", {}).get("probe_layers", 4),
        ignore_special_tokens=drift_cfg.get("drift_probe", {}).get("ignore_special_tokens", True),
        device=str(hydra_cfg.device),
    )
    s_norm = normalize_probe_score(s, drift_cfg)
    rr = rr_from_entropy(s_norm, drift_cfg)

    # slice fraction relative to base checkpoint rank
    frac = min(1.0, rr / BASE_RR)
    set_model_rank_frac(model, frac)

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