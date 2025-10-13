#coding:utf8
from typing import OrderedDict, Tuple
import warnings
warnings.filterwarnings("ignore", message=".*RoPE2D.*")
warnings.filterwarnings("ignore", message=".*version instead.*")
warnings.filterwarnings("ignore", message=".*_register_pytree_node is deprecated.*")
import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)
import pandas as pd
from contextlib import nullcontext
import argparse
import plotly.express as px
from safetensors.torch import load_file
from tqdm import tqdm
import torch
from accelerate import Accelerator
import torch.nn as nn
import csv
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
from pi3.models.layers.block import BlockRope
from SVD_LLM.component.svd_pi3 import SVD_Pi3Attention, SVD_Pi3MLP
from SVD_LLM.utils.data_utils import *
from SVD_LLM.utils.model_utils import *
from fvcore.nn import FlopCountAnalysis
import torch.profiler as prof
import random
from typing import List, Dict, Optional
from PIL import Image
from torchvision import transforms

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
RGB_MODALITIES = ["final"]

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
            if fn.lower().endswith(IMG_EXTS):
                frames.append(os.path.join(scene_dir, fn))
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
    sampling_stride: int = 1,
    split: str = "training",
    seed: int = 3,
    cache_dir: str = "/data/wanghaoxuan/SVD_Pi3_cache",
) -> List[Dict[str, torch.Tensor]]:
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir,
        f"sintel_{split}_ALLMODS_{nsamples}_{image_size}_{batch_size}_{sampling_stride}_{seed}.pt"
    )
    if os.path.exists(cache_file):
        print(f"[💥WARNING💥] Loading cached calibration data from {cache_file}...")
        return torch.load(cache_file)
    else:
        print(f"[NOTE] Building new calibration data and caching to {cache_file}...")

    random.seed(seed)
    torch.manual_seed(seed)

    # collect strided frames per modality
    frames_per_mod = {}
    for mod in RGB_MODALITIES:
        frames = collect_sintel_frames(root, split, mod)
        strided = frames[::max(1, sampling_stride)] or frames
        frames_per_mod[mod] = strided

    # allocate samples evenly (roughly) across modalities
    base = nsamples // len(RGB_MODALITIES)
    rem = nsamples % len(RGB_MODALITIES)
    per_mod_quota = {m: base + (i < rem) for i, m in enumerate(RGB_MODALITIES)}

    # sample frames per modality, with replacement if needed
    # sample frames per modality with exact-size guarantee
    chosen = []
    leftovers = []   # unused frames to backfill from
    deficit = 0

    for mod in RGB_MODALITIES:
        pool = frames_per_mod[mod]
        q = per_mod_quota[mod]

        if not pool:
            deficit += q
            continue

        if len(pool) >= q:
            picks = random.sample(pool, q)        # unique picks from this modality
            chosen.extend(picks)
            # keep remaining frames for possible backfill
            leftovers.extend([p for p in pool if p not in picks])
        else:
            # take all we have, note how many we still owe
            chosen.extend(pool)
            deficit += (q - len(pool))
            # no leftovers from this modality (exhausted)

    # backfill any shortfall from leftovers; if still short, from all frames
    if len(chosen) < nsamples:
        global_pool = leftovers if leftovers else sum(frames_per_mod.values(), [])
        while len(chosen) < nsamples and global_pool:
            chosen.append(random.choice(global_pool))

    # safety cap (shouldn’t trigger unless quotas changed upstream)
    if len(chosen) > nsamples:
        chosen = random.sample(chosen, nsamples)

    random.shuffle(chosen)

    # image preprocessing + batching
    to_tensor = build_transform(image_size=image_size, center_crop=True)
    traindataset: List[Dict[str, torch.Tensor]] = []
    batch_imgs: Optional[List[torch.Tensor]] = None
    for idx, path in enumerate(chosen):
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

    torch.save(traindataset, cache_file)
    return traindataset

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

@torch.no_grad()
def Pi3_profile_svdllm_low_resource(
    model: Pi3,
    calib_batches: List[Dict[str, torch.Tensor]],
    device: torch.device,
    autocast: bool = True,
    dtype: torch.dtype = torch.float16,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """
    Stream calibration data through Pi3 and compute per-module whitening factors.

    Returns
    -------
    profiling_mat: Dict[str, Tensor]
        Maps module_key -> Cholesky factor L of covariance (so that Cov ≈ L @ L^T).
    """
    model.eval().to(device)

    # choose targets (attention linear layers and MLP linear layers)
    targets = OrderedDict()
    # Pi3.decoder is nn.ModuleList[BlockRope]
    for i, blk in enumerate(model.decoder):
        blk: BlockRope = blk # type check
        # attention
        if hasattr(blk, "attn"):
            attn = blk.attn
            if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                targets[f"decoder.{i}.attn.qkv"] = attn.qkv
            if hasattr(attn, "proj") and isinstance(attn.proj, nn.Linear):
                targets[f"decoder.{i}.attn.proj"] = attn.proj
        # mlp (ffn)
        if hasattr(blk, "mlp"):
            mlp = blk.mlp
            if hasattr(mlp, "fc1") and isinstance(mlp.fc1, nn.Linear):
                targets[f"decoder.{i}.mlp.fc1"] = mlp.fc1
            if hasattr(mlp, "fc2") and isinstance(mlp.fc2, nn.Linear):
                targets[f"decoder.{i}.mlp.fc2"] = mlp.fc2
    
    print(f"✅Found {len(targets)} Linear targets in Pi3.decoder to whiten")

    # initialize per-module accumulators
    # G = X^T * X, N = total rows collected
    grams: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    for k, lin in targets.items():
        in_dim = lin.in_features
        grams[k] = torch.zeros((in_dim, in_dim), dtype=torch.float64, device=device)
        counts[k] = 0

    # define hooks to collect statistics during the calibration forward passes
    handles = []
    def make_pre_hook(key: str):
        def _hook(module: nn.Linear, inp):
            # inp is a tuple; grab first
            x = inp[0]
            # expected shapes: (..., in_features)
            x = x.detach()
            # collapse all leading dims to rows
            x = x.reshape(-1, x.shape[-1]).to(device, dtype=torch.float32)
            # center? (optional) — for calibration whitening, uncentered is okay
            G = x.T @ x   # (in_features, in_features) in float32
            grams[key] += G.to(torch.float64)
            counts[key] += x.shape[0]
        return _hook

    for key, lin in targets.items():
        lin: nn.Linear = lin  # type check
        handles.append(lin.register_forward_pre_hook(make_pre_hook(key)))
    print(f"✅Registered {len(handles)} forward hooks!")


    # run forward streaming through batches
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=dtype) if (autocast and device.type == "cuda")
        else nullcontext()
    )

    for b in tqdm(calib_batches):
        imgs = b["pixel_values"].to(device)
        # Pi3.forward expects (B, N, C, H, W). Our sampler returns (B, C, H, W).
        # Use N=1.
        imgs = imgs.unsqueeze(1)

        with amp_ctx:
            # here, PyTorch automatically invokes every registered forward_pre_hook
            _ = model(imgs)
    print(f"✅Completed streaming {len(calib_batches)} calibration batches")

    # remove hooks
    for h in handles:
        h.remove()
    print(f"✅Removed all {len(handles)} forward hooks")
    torch.cuda.synchronize(device) if device.type == "cuda" else None

    # build whitening matrices
    profiling_mat: Dict[str, torch.Tensor] = {}

    num_modules = len(targets)
    print(f"Building {num_modules} Cholesky factors (on CPU)...")
    fail_case = 0

    for key in tqdm(targets.keys()):
        n = max(1, counts[key])

        # 1) CPU float64 & symmetrize
        cov = (grams[key] / n).to(torch.float64).cpu()
        cov = 0.5 * (cov + cov.T)

        d = cov.shape[0]
        I = torch.eye(d, dtype=cov.dtype, device=cov.device)

        # scale-aware base shrinkage (Ledoit-Wolf style tiny alpha)
        mu = float(cov.trace() / max(1, d))
        base_eps = 1e-6 * max(1.0, mu)   # adapt to magnitude
        cov_j = cov + base_eps * I

        # 2) try Cholesky on CPU
        try:
            L = torch.linalg.cholesky(cov_j)
        except Exception:
            fail_case += 1
            evals, Q = torch.linalg.eigh(cov)  # CPU, float64, symmetric
            lam = torch.clamp(evals, min=base_eps)
            L = Q @ torch.diag(torch.sqrt(lam))

        profiling_mat[key] = L  # keep on CPU to save VRAM

    print(f"✅{num_modules - fail_case}/{num_modules} succeeded with Cholesky, {fail_case}/{num_modules} used EVD fallback")

    # clean up
    for k in grams:
        grams[k] = None
    torch.cuda.empty_cache()

    return profiling_mat


@torch.no_grad()
def Pi3_whitening(model: Pi3, profiling_mat: Dict[str, torch.Tensor], ratio: float, device=None):
    model.eval()
    
    # choose targets/layers (attention linear layers and MLP linear layers)
    layers = OrderedDict()
    # Pi3.decoder is nn.ModuleList[BlockRope]
    for i, blk in enumerate(model.decoder):
        blk: BlockRope = blk # type check
        # attention
        if hasattr(blk, "attn"):
            attn = blk.attn
            if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                layers[f"decoder.{i}.attn.qkv"] = attn.qkv
            if hasattr(attn, "proj") and isinstance(attn.proj, nn.Linear):
                layers[f"decoder.{i}.attn.proj"] = attn.proj
        # mlp (ffn)
        if hasattr(blk, "mlp"):
            mlp = blk.mlp
            if hasattr(mlp, "fc1") and isinstance(mlp.fc1, nn.Linear):
                layers[f"decoder.{i}.mlp.fc1"] = mlp.fc1
            if hasattr(mlp, "fc2") and isinstance(mlp.fc2, nn.Linear):
                layers[f"decoder.{i}.mlp.fc2"] = mlp.fc2
    print(f"Start SVD decomposition after whitening {len(layers)} Linear targets...")

    # Caches for per-block replacements
    svd_attn_cache: Dict[int, SVD_Pi3Attention] = {}
    svd_mlp_cache: Dict[int, SVD_Pi3MLP] = {}
    
    def ensure_svd_attn(block_idx: int, orig_attn) -> SVD_Pi3Attention:
        if block_idx in svd_attn_cache:
            return svd_attn_cache[block_idx]
        D = orig_attn.qkv.in_features
        H = getattr(orig_attn, "num_heads", 16)  # fallback if not present
        # Derive ranks from the two actual matrices to be factorized when we have them
        # We'll set temporary ranks (updated after we know per-matrix r below)
        r_qkv = max(1, D // 4)
        r_out = max(1, D // 4)
        attn_drop = getattr(orig_attn, "attn_drop", 0.0)
        proj_drop = getattr(orig_attn, "proj_drop", 0.0)
        rope = getattr(orig_attn, "rope", None)

        svd_attn = SVD_Pi3Attention(
            embed_dim=D, num_heads=H,
            r_qkv=r_qkv, r_out=r_out,
            attn_drop_rate=attn_drop, proj_drop_rate=proj_drop,
            use_bias_qkv=(orig_attn.qkv.bias is not None),
            use_bias_out=(orig_attn.proj.bias is not None),
            rope=rope
        )
        # install
        model.decoder[block_idx].attn = svd_attn
        svd_attn_cache[block_idx] = svd_attn
        return svd_attn

    def ensure_svd_mlp(block_idx: int, orig_mlp) -> SVD_Pi3MLP:
        if block_idx in svd_mlp_cache:
            return svd_mlp_cache[block_idx]
        D = orig_mlp.fc1.in_features
        I = orig_mlp.fc1.out_features
        # temporary ranks; will be overwritten by actual per-matrix r below
        r1 = max(1, (I + D) // 8)
        r2 = max(1, (I + D) // 8)

        # try to detect activation
        act_name = "gelu"
        if hasattr(orig_mlp, "act") and isinstance(orig_mlp.act, nn.Module):
            act_name = orig_mlp.act.__class__.__name__.lower()

        drop = getattr(orig_mlp, "drop", 0.0)

        svd_mlp = SVD_Pi3MLP(
            embed_dim=D, intermediate_dim=I,
            r_fc1=r1, r_fc2=r2,
            activation=act_name, drop_rate=drop,
            use_bias_fc1=(orig_mlp.fc1.bias is not None),
            use_bias_fc2=(orig_mlp.fc2.bias is not None),
        )
        model.decoder[block_idx].mlp = svd_mlp
        svd_mlp_cache[block_idx] = svd_mlp
        return svd_mlp

    def whiten_svd_factor(
        key: str,
        linear: nn.Linear,
        profiling_mat: Dict[str, torch.Tensor],
        ratio: float,
        dev: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Robust whitening + SVD factorization for a single Linear layer.
        Returns (svd_u, svd_v, bias), with balanced sqrt(S) split.
        Handles:
        - vector/diag and full scale matrices
        - singular / ill-conditioned scales via clamping
        - non-finite values via sanitization
        - SVD solver issues via dtype/CPU fallback and PCA fallback
        """

        def trunc_rank(m: int, n: int, r: float) -> int:
            rr = int((m * n * r) / (m + n))
            return max(1, min(rr, min(m, n)))

        def is_diag_matrix(M: torch.Tensor) -> bool:
            return (
                M.dim() == 2 and M.shape[0] == M.shape[1] and
                torch.allclose(M, torch.diag(torch.diagonal(M)), atol=1e-6, rtol=0)
            )

        def sanitize(t: torch.Tensor, replace: float = 0.0) -> torch.Tensor:
            # Replace NaN/Inf with finite values
            t = t.clone()
            mask = ~torch.isfinite(t)
            if mask.any():
                t[mask] = replace
            return t

        def add_jitter(M: torch.Tensor, scale: float = 1e-8) -> torch.Tensor:
            # For rectangular M, add tiny Gaussian noise scaled by Fro norm
            fro = torch.linalg.norm(M).item()
            eps = scale * (fro if fro > 0 else 1.0)
            noise = torch.empty_like(M).normal_(mean=0.0, std=eps)
            return M + noise

        def safe_svd(M: torch.Tensor):
            # Try FP32 on device
            try:
                return torch.linalg.svd(M, full_matrices=False)
            except Exception:
                pass
            # Try FP64 on device
            try:
                return torch.linalg.svd(M.to(torch.float64), full_matrices=False)
            except Exception:
                pass
            # Try CPU FP64
            try:
                U, S, VT = torch.linalg.svd(M.detach().cpu().to(torch.float64), full_matrices=False)
                return U.to(M.device, dtype=M.dtype), S.to(M.device, dtype=M.dtype), VT.to(M.device, dtype=M.dtype)
            except Exception:
                raise RuntimeError("SVD failed on device and CPU in both float32 and float64. Bad times!")

        W = linear.weight.data.float().to(dev)          # (out, in)
        W = sanitize(W)
        dtype = linear.weight.dtype

        Sraw = profiling_mat[key].to(dev).float()
        Sraw = sanitize(Sraw)

        out_dim, in_dim = W.shape
        eps_abs = 1e-6
        eps_rel = 1e-4

        if Sraw.dim() == 1 or is_diag_matrix(Sraw):
            # ----- diagonal / vector path -----
            s = Sraw if Sraw.dim() == 1 else torch.diagonal(Sraw)   # (in,)
            s = sanitize(s)
            s_absmax = s.abs().max()
            floor = max(eps_abs, float(eps_rel) * float(s_absmax)) if s_absmax > 0 else eps_abs
            s_safe = torch.clamp(s, min=floor)

            # W_scale = W @ diag(s)  (column-wise scaling)
            W_scale = W * s_safe.unsqueeze(0)
            W_scale = sanitize(W_scale)
            if not torch.isfinite(W_scale).all():
                W_scale = add_jitter(W_scale, scale=1e-8)

            U, Svals, VT = safe_svd(W_scale)

            r = trunc_rank(out_dim, in_dim, ratio)
            r = min(r, U.shape[1], VT.shape[0], Svals.shape[0])
            U_r, S_r, VT_r = U[:, :r], Svals[:r], VT[:r, :]

            inv_s = 1.0 / s_safe
            V_r = VT_r * inv_s.unsqueeze(0)  # undo whitening on V

        else:
            # ----- full matrix path (symmetric EVD + clamping) -----
            S = Sraw
            if S.shape != (in_dim, in_dim):
                raise ValueError(f"Scale matrix shape {S.shape} != ({in_dim},{in_dim})")

            # Symmetrize & sanitize
            S_sym = 0.5 * (S + S.transpose(-1, -2))
            S_sym = sanitize(S_sym)

            # eigh with fallback
            try:
                evals, Q = torch.linalg.eigh(S_sym)
            except Exception:
                evals, Q = torch.linalg.eigh(S_sym.detach().cpu().to(torch.float64))
                evals = evals.to(dev).to(torch.float32)
                Q = Q.to(dev).to(torch.float32)

            evals = sanitize(evals)
            Q = sanitize(Q)
            lam_max = evals.abs().max()
            floor = max(eps_abs, float(eps_rel) * float(lam_max)) if lam_max > 0 else eps_abs
            lam = torch.clamp(evals, min=floor)  # (in,)

            # Apply S on input side efficiently:
            # W_scale = (W @ Q) * lam  @ Q^T
            WQ = sanitize(W @ Q)
            W_scale = sanitize((WQ * lam.unsqueeze(0)) @ Q.transpose(-1, -2))
            if not torch.isfinite(W_scale).all():
                W_scale = add_jitter(W_scale, scale=1e-8)

            U, Svals, VT = safe_svd(W_scale)

            r = trunc_rank(out_dim, in_dim, ratio)
            r = min(r, U.shape[1], VT.shape[0], Svals.shape[0])
            U_r, S_r, VT_r = U[:, :r], Svals[:r], VT[:r, :]

            # Undo whitening on V via eigen-structure: Sinv = Q diag(1/lam) Q^T
            inv_lam = 1.0 / lam
            VTQ = sanitize(VT_r @ Q)
            V_r = sanitize((VTQ * inv_lam.unsqueeze(0)) @ Q.transpose(-1, -2))
        S_r = sanitize(S_r)
        sqrtS = torch.sqrt(torch.clamp(S_r, min=0.0) + 1e-12)
        svd_u = U_r * sqrtS.unsqueeze(0)     # (out, r)
        svd_v = sqrtS.unsqueeze(1) * V_r     # (r, in)
        svd_u = svd_u.detach().cpu().to(dtype)
        svd_v = svd_v.detach().cpu().to(dtype)
        bias = linear.bias.detach().cpu().to(dtype) if linear.bias is not None else None
        assert svd_u.shape[0] == out_dim, f"svd_u rows {svd_u.shape[0]} != out {out_dim}"
        assert svd_v.shape[1] == in_dim, f"svd_v cols {svd_v.shape[1]} != in {in_dim}"
        return svd_u, svd_v, bias

    for key, linear in tqdm(layers.items()):
        parts = key.split(".")  # ["decoder", "{i}", "attn"/"mlp", leaf]
        i = int(parts[1])
        sub = parts[2]
        leaf = parts[3]

        svd_u, svd_v, bias = whiten_svd_factor(key, linear, profiling_mat, ratio, dev)

        if sub == "attn":
            orig_attn = getattr(model.decoder[i], "attn")
            svd_attn = ensure_svd_attn(i, orig_attn)

            if leaf == "qkv":
                # update ranks to match actual factor dims
                svd_attn.qkv_u = nn.Linear(svd_v.shape[0], 3 * svd_attn.embed_dim, bias=(svd_attn.qkv_u.bias is not None))
                svd_attn.qkv_v = nn.Linear(svd_attn.embed_dim, svd_v.shape[0], bias=False)
                svd_attn.qkv_u.weight.data.copy_(svd_u)
                svd_attn.qkv_v.weight.data.copy_(svd_v)
                if svd_attn.qkv_u.bias is not None and bias is not None:
                    svd_attn.qkv_u.bias.data.copy_(bias)

            elif leaf == "proj":
                svd_attn.o_u = nn.Linear(svd_v.shape[0], svd_attn.embed_dim, bias=(svd_attn.o_u.bias is not None))
                svd_attn.o_v = nn.Linear(svd_attn.embed_dim, svd_v.shape[0], bias=False)
                svd_attn.o_u.weight.data.copy_(svd_u)
                svd_attn.o_v.weight.data.copy_(svd_v)
                if svd_attn.o_u.bias is not None and bias is not None:
                    svd_attn.o_u.bias.data.copy_(bias)

        elif sub == "mlp":
            svd_mlp = ensure_svd_mlp(i, getattr(model.decoder[i], "mlp"))

            in_f  = linear.in_features
            out_f = linear.out_features
            has_bias = (linear.bias is not None)

            if leaf == "fc1":
                # rebuild with dims taken from the *current* linear
                svd_mlp.fc1_u = nn.Linear(svd_v.shape[0], out_f, bias=has_bias)
                svd_mlp.fc1_v = nn.Linear(in_f,            svd_v.shape[0], bias=False)
                svd_mlp.fc1_u.weight.data.copy_(svd_u)
                svd_mlp.fc1_v.weight.data.copy_(svd_v)
                if has_bias:
                    svd_mlp.fc1_u.bias.data.copy_(bias)

            elif leaf == "fc2":
                svd_mlp.fc2_u = nn.Linear(svd_v.shape[0], out_f, bias=has_bias)
                svd_mlp.fc2_v = nn.Linear(in_f,            svd_v.shape[0], bias=False)
                svd_mlp.fc2_u.weight.data.copy_(svd_u)
                svd_mlp.fc2_v.weight.data.copy_(svd_v)
                if has_bias:
                    svd_mlp.fc2_u.bias.data.copy_(bias)

                del svd_u, svd_v, bias
                torch.cuda.empty_cache()

    print(f"✅ Pi3 whitening + SVD low-rank replacement complete for {len(layers)} Linear layers.")


@torch.no_grad()
def Pi3_svd_baseline(model: Pi3, ratio: float, device=None):
    """
    Baseline: directly apply SVD to each target Linear weight W (no whitening).
    Replaces attention.qkv / attention.proj / mlp.fc1 / mlp.fc2 with low-rank
    factorizations using the same SVD_Pi3Attention/SVD_Pi3MLP shells you already use.
    """
    model.eval()
    dev = torch.device(device) if device is not None else next(model.parameters()).device

    layers = OrderedDict()
    for i, blk in enumerate(model.decoder):
        # attention
        if hasattr(blk, "attn"):
            attn = blk.attn
            if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                layers[f"decoder.{i}.attn.qkv"] = attn.qkv
            if hasattr(attn, "proj") and isinstance(attn.proj, nn.Linear):
                layers[f"decoder.{i}.attn.proj"] = attn.proj
        # mlp
        if hasattr(blk, "mlp"):
            mlp = blk.mlp
            if hasattr(mlp, "fc1") and isinstance(mlp.fc1, nn.Linear):
                layers[f"decoder.{i}.mlp.fc1"] = mlp.fc1
            if hasattr(mlp, "fc2") and isinstance(mlp.fc2, nn.Linear):
                layers[f"decoder.{i}.mlp.fc2"] = mlp.fc2

    print(f"Start plain SVD (no whitening) for {len(layers)} Linear targets...")

    svd_attn_cache: Dict[int, SVD_Pi3Attention] = {}
    svd_mlp_cache: Dict[int, SVD_Pi3MLP] = {}

    def ensure_svd_attn(block_idx: int, orig_attn) -> SVD_Pi3Attention:
        if block_idx in svd_attn_cache:
            return svd_attn_cache[block_idx]
        D = orig_attn.qkv.in_features
        H = getattr(orig_attn, "num_heads", 16)
        r_qkv = max(1, D // 4)  
        r_out = max(1, D // 4)
        attn_drop = getattr(orig_attn, "attn_drop", 0.0)
        proj_drop = getattr(orig_attn, "proj_drop", 0.0)
        rope = getattr(orig_attn, "rope", None)
        svd_attn = SVD_Pi3Attention(
            embed_dim=D, num_heads=H,
            r_qkv=r_qkv, r_out=r_out,
            attn_drop_rate=attn_drop, proj_drop_rate=proj_drop,
            use_bias_qkv=(orig_attn.qkv.bias is not None),
            use_bias_out=(orig_attn.proj.bias is not None),
            rope=rope
        )
        model.decoder[block_idx].attn = svd_attn
        svd_attn_cache[block_idx] = svd_attn
        return svd_attn

    def ensure_svd_mlp(block_idx: int, orig_mlp) -> SVD_Pi3MLP:
        if block_idx in svd_mlp_cache:
            return svd_mlp_cache[block_idx]
        D = orig_mlp.fc1.in_features
        I = orig_mlp.fc1.out_features
        r1 = max(1, (I + D) // 8)  # temp; actual r set by factors below
        r2 = max(1, (I + D) // 8)
        act_name = "gelu"
        if hasattr(orig_mlp, "act") and isinstance(orig_mlp.act, nn.Module):
            act_name = orig_mlp.act.__class__.__name__.lower()
        drop = getattr(orig_mlp, "drop", 0.0)
        svd_mlp = SVD_Pi3MLP(
            embed_dim=D, intermediate_dim=I,
            r_fc1=r1, r_fc2=r2,
            activation=act_name, drop_rate=drop,
            use_bias_fc1=(orig_mlp.fc1.bias is not None),
            use_bias_fc2=(orig_mlp.fc2.bias is not None),
        )
        model.decoder[block_idx].mlp = svd_mlp
        svd_mlp_cache[block_idx] = svd_mlp
        return svd_mlp

    def trunc_rank(m: int, n: int, r: float) -> int:
        # same heuristic as your whitening version
        rr = int((m * n * r) / (m + n))
        return max(1, min(rr, min(m, n)))

    def sanitize(t: torch.Tensor, replace: float = 0.0) -> torch.Tensor:
        t = t.clone()
        mask = ~torch.isfinite(t)
        if mask.any():
            t[mask] = replace
        return t

    def safe_svd(M: torch.Tensor):
        try:
            return torch.linalg.svd(M, full_matrices=False)
        except Exception:
            pass
        try:
            return torch.linalg.svd(M.to(torch.float64), full_matrices=False)
        except Exception:
            pass
        try:
            U, S, VT = torch.linalg.svd(M.detach().cpu().to(torch.float64), full_matrices=False)
            return U.to(M.device, dtype=M.dtype), S.to(M.device, dtype=M.dtype), VT.to(M.device, dtype=M.dtype)
        except Exception:
            raise RuntimeError("SVD failed on device and CPU in both float32 and float64.")

    for key, linear in tqdm(layers.items()):
        parts = key.split(".")
        i = int(parts[1])
        sub = parts[2]
        leaf = parts[3]

        # W only (no whitening)
        W = linear.weight.data.to(dev).float()
        W = sanitize(W)

        out_dim, in_dim = W.shape
        U, Svals, VT = safe_svd(W)

        r = trunc_rank(out_dim, in_dim, ratio)
        r = min(r, U.shape[1], VT.shape[0], Svals.shape[0])

        U_r, S_r, VT_r = U[:, :r], Svals[:r], VT[:r, :]  # (out,r), (r,), (r,in)

        # balanced split
        S_r = sanitize(S_r)
        sqrtS = torch.sqrt(torch.clamp(S_r, min=0.0) + 1e-12)
        svd_u = (U_r * sqrtS.unsqueeze(0)).detach().cpu().to(linear.weight.dtype)  # (out,r)
        svd_v = (sqrtS.unsqueeze(1) * VT_r).detach().cpu().to(linear.weight.dtype) # (r,in)
        bias = linear.bias.detach().cpu().to(linear.weight.dtype) if linear.bias is not None else None

        # ----- install into shells -----
        if sub == "attn":
            orig_attn = getattr(model.decoder[i], "attn")
            svd_attn = ensure_svd_attn(i, orig_attn)

            if leaf == "qkv":
                # qkv: out = 3*D, in = D
                svd_attn.qkv_u = nn.Linear(svd_v.shape[0], 3 * svd_attn.embed_dim, bias=(svd_attn.qkv_u.bias is not None))
                svd_attn.qkv_v = nn.Linear(svd_attn.embed_dim, svd_v.shape[0], bias=False)
                svd_attn.qkv_u.weight.data.copy_(svd_u)  # (3D, r)
                svd_attn.qkv_v.weight.data.copy_(svd_v)  # (r, D)
                if svd_attn.qkv_u.bias is not None and bias is not None:
                    svd_attn.qkv_u.bias.data.copy_(bias)

            elif leaf == "proj":
                svd_attn.o_u = nn.Linear(svd_v.shape[0], svd_attn.embed_dim, bias=(svd_attn.o_u.bias is not None))
                svd_attn.o_v = nn.Linear(svd_attn.embed_dim, svd_v.shape[0], bias=False)
                svd_attn.o_u.weight.data.copy_(svd_u)  # (D, r)
                svd_attn.o_v.weight.data.copy_(svd_v)  # (r, D)
                if svd_attn.o_u.bias is not None and bias is not None:
                    svd_attn.o_u.bias.data.copy_(bias)

        elif sub == "mlp":
            svd_mlp = ensure_svd_mlp(i, getattr(model.decoder[i], "mlp"))

            in_f  = linear.in_features
            out_f = linear.out_features
            has_bias = (linear.bias is not None)

            if leaf == "fc1":
                svd_mlp.fc1_u = nn.Linear(svd_v.shape[0], out_f, bias=has_bias)
                svd_mlp.fc1_v = nn.Linear(in_f,            svd_v.shape[0], bias=False)
                svd_mlp.fc1_u.weight.data.copy_(svd_u)  # (out_f, r)
                svd_mlp.fc1_v.weight.data.copy_(svd_v)  # (r, in_f)
                if has_bias and bias is not None:
                    svd_mlp.fc1_u.bias.data.copy_(bias)

            elif leaf == "fc2":
                svd_mlp.fc2_u = nn.Linear(svd_v.shape[0], out_f, bias=has_bias)
                svd_mlp.fc2_v = nn.Linear(in_f,            svd_v.shape[0], bias=False)
                svd_mlp.fc2_u.weight.data.copy_(svd_u)
                svd_mlp.fc2_v.weight.data.copy_(svd_v)
                if has_bias and bias is not None:
                    svd_mlp.fc2_u.bias.data.copy_(bias)

                del svd_u, svd_v, bias
                torch.cuda.empty_cache()

    print(f"✅ Plain SVD (no whitening) low-rank replacement complete for {len(layers)} Linear layers.")

def main():
    # NOTE: whether to run the baseline SVD (no whitening) mode
    BASELINE = False

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='jeffwan/llama-7b-hf', help='LLaMA model to load, pass `jeffwan/llama-7b-hf`')
    parser.add_argument('--model_path', type=str, default=None, help='local compressed model path or whitening information path')
    parser.add_argument('--run_low_resource', action='store_true', help='whether to run whitening in low resource, exp, compress LLaMA-7B below 15G gpu')
    parser.add_argument('--dataset', type=str, default='wikitext2',help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Number of calibration data samples for whitening.')
    parser.add_argument('--updating_nsamples', type=int, default=16, help='Number of calibration data samples for udpating.')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save the compressed model checkpoints.`')
    parser.add_argument('--seed',type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--DEV', type=str, default="cuda", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='inference bactch size')
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len for efficiency evaluation')
    parser.add_argument('--step', type=int, default=4, help='the step to run the compression')
    parser.add_argument('--lora', type=str, default=None, help='the lora updated weight path to run the accuracy evaluation')
    parser.add_argument("--interval", type=int, default=-1, help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to the model checkpoint file. Default: None")
    parser.add_argument('--ratio', type=float, default=0.2, help='Target compression ratio,(0,1), default=0.2, means only keeping about 20% of the params.')
    parser.add_argument("--device", type=str, default='cuda', help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    parser.add_argument("--calibration_dataset_path", type=str, default="/data/wanghaoxuan/sintel", help="Path to the calibration dataset.")

    args = parser.parse_args()
    args.ratio = 1- args.ratio

    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3().to(device).eval()
        if args.ckpt.endswith('.safetensors'): 
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        
        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()

    model = model.eval()
    print("✅ model loaded.")


    if not BASELINE:
        # collect calibration data
        print("Start collecting calibration data...")
        cali_white_data = Pi3_get_calib_train_data(
            root=args.calibration_dataset_path,
            nsamples=args.whitening_nsamples
        )
        print(f"✅ collected {len(cali_white_data)} calibration batches (~{sum(b['pixel_values'].shape[0] for b in cali_white_data)} images).")

        # derive the whitening matrix via profiling
        profiling_mat = Pi3_profile_svdllm_low_resource(model, cali_white_data, device, autocast=True, dtype=torch.float16, eps=1e-6)

        # apply whitening
        Pi3_whitening(model, profiling_mat, args.ratio, args.DEV)

        # save the model using accelerate
        accelerator = Accelerator()
        state_dict = accelerator.get_state_dict(model)
        from safetensors.torch import save_file
        out_path = f"{args.save_path}/Pi3_whitening_only_{str(args.ratio)}.safetensors"
        save_file(state_dict, out_path)
    else:
        Pi3_svd_baseline(model, args.ratio, args.DEV)
        # save the model using accelerate
        accelerator = Accelerator()
        state_dict = accelerator.get_state_dict(model)
        from safetensors.torch import save_file
        out_path = f"{args.save_path}/Pi3_svd_baseline_{str(args.ratio)}.safetensors"
        save_file(state_dict, out_path)    
    print("✅✅✅ALL DONE!✅✅✅")

if __name__ == "__main__":
    main()