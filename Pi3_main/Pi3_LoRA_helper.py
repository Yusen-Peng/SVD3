from logging import config
import os
import sys
import argparse
from easydict import EasyDict
from typing import List

import hydra
import torch
import transformers
from datasets import load_dataset
from safetensors.torch import load_file
from omegaconf import OmegaConf
from omegaconf import DictConfig

from pi3.models.pi3 import Pi3

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)
from utils.peft import LoraConfig, get_peft_model
from SVD_LLM.utils.data_utils import *
from SVD_LLM.utils.model_utils import *
from SVD_LLM.evaluater import *

class TwoFactorLinear(nn.Module):
    def __init__(self, in_features, out_features, r, has_bias):
        super().__init__()
        self.v = nn.Linear(in_features, r, bias=False)
        self.u = nn.Linear(r, out_features, bias=has_bias)
    def forward(self, x):
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
                old = getattr(parent, leaf_name)  # original nn.Linear

                in_f, out_f = old.in_features, old.out_features
                r = sd[k_v_w].shape[0]
                has_bias = (k_u_b in sd)

                # Build TwoFactorLinear with correct geometry
                tfl = TwoFactorLinear(in_features=in_f, out_features=out_f, r=r, has_bias=has_bias)
                tfl = tfl.to(device=old.weight.device, dtype=old.weight.dtype)

                setattr(parent, leaf_name, tfl)
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"

# separate configs for SEQUENTIAL updates
# U-phase = LoRA on U linears
U_TARGETS = [
    "attn.qkv.u",
    "attn.proj.u",
    "mlp.fc1.u",
    "mlp.fc2.u",
]

# V-phase = LoRA on V linears
V_TARGETS = [
    "attn.qkv.v",
    "attn.proj.v",
    "mlp.fc1.v",
    "mlp.fc2.v",
]


def build_pi3_with_lora(ckpt_path: str, device: torch.device, *,
                        phase: str, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    """
    load a whitened Pi3, and wrap with LoRA
    phase: "U" or "V"
    returns model (nn.Module), and info dict
    """
    sd = load_file(ckpt_path, device=str(device))
    
    print(f"😎Loading the compressed Pi3 from {ckpt_path}...")
    model = Pi3().to(device).eval()
    install_twofactor_modules_from_sd(model, sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print("Note: unexpected keys (benign):", unexpected)
    if missing:
        print("Note: missing keys (benign if non-decoder):", missing)

    model.to(device)

    targets = U_TARGETS if phase.upper()=="U" else V_TARGETS
    lc = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=targets,
        bias="none",
        task_type=None,
        fan_in_fan_out=False,
    )

    model = get_peft_model(model, lc) # wrap with LoRA
    return model, {"phase": phase.upper(), "targets": targets, "r": r, "alpha": alpha, "dropout": dropout}


def build_cfg(args, phase: str, ckpt_path: str, out_dir: str) -> DictConfig:
    """
    Build a minimal DictConfig the Pi3TrainerLoRA stack expects, without Hydra CLI.
    - phase: "U", "V", or "NONE"
    - ckpt_path: path to your whitening-only CompressedPi3 checkpoint
    - out_dir: where logs/ckpts go
    """
    cfg = {
        "random_seed": 42,

        # Minimal logging config so get_logger() won’t crash when Hydra is not used
        "job_logging_cfg": {
            "version": 1,
            "formatters": {"simple": {"format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s"}},
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout",
                }
            },
            "root": {"level": "INFO", "handlers": ["console"]},
            "disable_existing_loggers": False,
        },

        # Model & LoRA
        "model": {
            "ckpt": ckpt_path,
        },
        "lora": {
            "phase": phase, # "U" then "V", two steps of sequential
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },

        # Training knobs (BaseTrainer / create_dataloader read from here)
        "train": {
            "num_epoch": args.num_epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": max(1, args.micro_batch_size),
            "optimizer": {
                "type": "AdamW",
                "lr": args.learning_rate,
                "encoder_lr": args.learning_rate,
                "weight_decay": 0.01,
            },
            "lr_scheduler": {
                "type": "CosineAnnealingLR",
                "total_steps": 0,  # NOTE: just a placeholder, set by Pi3TrainerLoRA
            },

            "print_freq": 50,
            "model_dtype": "fp32",
            "clip_grad": 1.0,
            "clip_loss": 1e4,
            "base_seed": 42,
            "find_unused_parameters": False,
            "resume": None,
            "iters_per_epoch": 0,
            "image_num_range": [8, 8],        # frames per sample
            "num_resolution": 1,
            "random_reslution": False,        # (typo preserved from upstream)
            "patch_size": 14,
            "aspect_ratio_range": [0.5, 2.0],
            "pixel_count_range": [224*224, 224*224],
            "max_img_per_gpu": 8,
            "num_workers": 8,                 # used below by create_dataloader
        },
        "test": {
            "batch_size": args.batch_size,
            "num_workers": 8,
            "iters_per_test": 0,
        },

        "log": {
            "use_wandb": False,
            "use_tensorboard": False,
            "output_dir": out_dir,
            "ckpt_dir": f"{out_dir}/ckpts",
            "ckpt_interval": 1,
            "max_checkpoints": 3,
        },
        "train_dataloader": {"drop_last": True, "shuffle": True},
        "test_dataloader":  {"drop_last": False, "shuffle": False},
        "train_dataset": {
            "weights": {"CO3DV2": 10000},
            "CO3DV2": {
                "_target_": "local_datasets.co3dv2_dataset.CO3DV2Dataset",
                "data_root": "/data/wanghaoxuan/CO3Dv2_single_seq",
                "frame_num": 8,
                "mode": "train",
                "aug_crop": 16,
                "aug_focal": 0.9,
                "resolution": [224, 224],
                "transform": {
                    "_partial_": True,
                    "_target_": "local_datasets.base.transforms.JitterJpegLossBlurring",
                },
            },
        },
        "test_dataset": {
            "weights": {"CO3DV2": 10000},
            "CO3DV2": {
                "_target_": "local_datasets.co3dv2_dataset.CO3DV2Dataset",
                "data_root": "/data/wanghaoxuan/CO3Dv2_single_seq",
                "frame_num": 8,
                "mode": "test",
                "aug_crop": 16,
                "aug_focal": 0.9,
                "resolution": [224, 244], # subject to change
                "transform": {
                    "_partial_": True,
                    "_target_": "local_datasets.base.transforms.JitterJpegLossBlurring",
                },
            },
        },


        "loss": {
            "train_loss": {
                "_target_": "pi3.models.loss.Pi3Loss",
            }
        },

        # Which trainer to instantiate
        "trainer": "Pi3TrainerLoRA",
    }
    return OmegaConf.create(cfg)
