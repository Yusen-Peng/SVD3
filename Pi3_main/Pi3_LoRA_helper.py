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

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)
from utils.peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from utils.Prompter import Prompter, ZeroPrompter
from pi3.models.pi3 import CompressedPi3
from SVD_LLM.utils.data_utils import *
from SVD_LLM.utils.model_utils import *
from SVD_LLM.evaluater import *

device = "cuda" if torch.cuda.is_available() else "cpu"

# separate configs for SEQUENTIAL updates
U_TARGETS = ["qkv_u", "o_u", "fc1_u", "fc2_u"]
V_TARGETS = ["qkv_v", "o_v", "fc1_v", "fc2_v"]

# ======================== wrap the model with PEFT =========================
# ============================================================================
def build_pi3_with_lora(ckpt_path: str, device: torch.device, *,
                        phase: str, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    """
    load a whitened Pi3, and wrap with LoRA
    phase: "U", "V", or "NONE"
    returns model (nn.Module), and info dict
    """
    model = CompressedPi3().to(device).eval()
    sd = load_file(ckpt_path, device="cpu")
    model.load_factorized_state_dict(sd, strict=True)

    if phase == "NONE":
        return model, {"phase": "NONE"}

    targets = U_TARGETS if phase.upper()=="U" else V_TARGETS
    lc = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=targets,
        bias="none",
        task_type=None,   # keep unmapped → generic PEFT path
        fan_in_fan_out=False,
    )
    model = get_peft_model(model, lc)   # wraps with PeftModel→LoraModel
    return model, {"phase": phase.upper(), "targets": targets, "r": r, "alpha": alpha, "dropout": dropout}


# ======================== main finetuning configs =========================
# ==========================================================================
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
            "ckpt": ckpt_path,  # used by Pi3TrainerLoRA.prepare_model -> build_pi3_with_lora
        },
        "lora": {
            "phase": phase,                 # "U" then "V"
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },

        # Training knobs (BaseTrainer / create_dataloader read from here)
        "train": {
            "num_epoch": args.num_epochs,
            "batch_size": args.batch_size,
            # if you want true gradient accumulation set this explicitly; otherwise 1 is fine
            "gradient_accumulation_steps": max(1, args.micro_batch_size),

            "optimizer": {
                "lr": args.learning_rate,
                "encoder_lr": args.learning_rate,
                "weight_decay": 0.01,
            },
            "lr_scheduler": {
                "name": "cosine",
                "warmup_steps": 100,
                "total_steps": 0,  # filled by trainer after it knows dataloader length
            },

            "print_freq": 50,
            "model_dtype": "fp16",        # "no", "fp16", or "bf16"
            "clip_grad": 1.0,
            "clip_loss": 1e4,
            "base_seed": 42,
            "find_unused_parameters": False,

            # fields consumed by create_dataloader / samplers
            "iters_per_epoch": 0,             # 0 => use len(dataloader)
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
            "batch_size": 2,
            "num_workers": 4,
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

        # Dataloader flags
        "train_dataloader": {"drop_last": True, "shuffle": True},
        "test_dataloader":  {"drop_last": False, "shuffle": False},

        # === DATASETS ===
        # If your dataset classes live under a different module, change _target_ accordingly.
        "train_dataset": {
            "weights": {"Sintel": 1000},
            "Sintel": {
                "_target_": "datasets.sintel_dataset.SintelDataset",  # <-- adjust to your repo
                "root_path": "/data/wanghaoxuan/sintel/training",
                "frame_num": 8,
                "mode": "train",
                "aug_crop": 16,
                "aug_focal": 0.9,
                "transform": {
                    "_partial_": True,
                    "_target_": "datasets.base.transforms.JitterJpegLossBlurring",  # <-- adjust if using local_datasets.*
                },
                # "resolution": [[224, 224]],  # optionally pin a single res
            },
        },
        "test_dataset": {
            "weights": {"Sintel": 200},
            "Sintel": {
                "_target_": "datasets.sintel_dataset.SintelDataset",  # <-- adjust to your repo
                "root_path": "/data/wanghaoxuan/sintel/training",
                "frame_num": 8,
                "mode": "test",
                "transform": {
                    "_partial_": True,
                    "_target_": "datasets.base.transforms.ImgToTensor",  # <-- adjust if using local_datasets.*
                },
                "resolution": [[224, 224]],
            },
        },

        # Loss (make sure import path matches your tree)
        "loss": {
            "train_loss": {
                "_target_": "pi3.models.loss.Pi3Loss",
            }
        },

        # Which trainer to instantiate
        "trainer": "Pi3TrainerLoRA",
    }
    return OmegaConf.create(cfg)