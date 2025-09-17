'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''

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

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)
from trainers.pi3_trainer import Pi3TrainerLoRA
from SVD_LLM.utils.peft.utils.config import TaskType
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






def build_cfg(args, phase, ckpt_in, out_dir):
    grad_acc = max(1, args.batch_size // args.micro_batch_size)
    return EasyDict({
        "model": {"ckpt": ckpt_in},
        "lora":  {"phase": phase, "r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "dataset": {
            # inline your dataset block exactly like your YAML,
            # or import & load one of your YAMLs and merge here.
            "sintel": {
                "type": "video",
                "root_path": "/data/wanghaoxuan/sintel/training",
                "ls_all_seqs": ["alley_2","ambush_4","ambush_5","ambush_6",
                                "cave_2","cave_4","market_2","market_5","market_6",
                                "shaman_3","sleeping_1","sleeping_2","temple_2","temple_3"],
                "img":   {"path": "${data.sintel.root_path}/final/{seq}", "ext": "png"},
                "depth": {"path": "${data.sintel.root_path}/depth/{seq}", "ext": "dpt"},
            }
        },
        "train": {
            "num_epoch": args.num_epochs,
            "batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": grad_acc,
            "model_dtype": "fp16",
            "optimizer": EasyDict({"lr": args.learning_rate, "encoder_lr": args.learning_rate, "weight_decay": 0.0}),
            "lr_scheduler": EasyDict({"total_steps": 0}),
            "iters_per_epoch": 0,
            "clip_grad": 1.0,
            "clip_loss": 1e6,
            "base_seed": 42,
            "find_unused_parameters": False,
            "resume": None,
        },
        "test": {"iters_per_test": 0},
        "log":  {"output_dir": out_dir, "ckpt_dir": os.path.join(out_dir, "ckpts"),
                 "max_checkpoints": 3, "use_wandb": False, "use_tensorboard": True},
        "loss": {"train_loss": {"_target_": "pi3.models.loss.Pi3Loss"}},
    })


def run_phase(args, phase, ckpt_in, out_dir):
    cfg = build_cfg(args, phase, ckpt_in, out_dir)
    trainer = Pi3TrainerLoRA(cfg)
    trainer.train()
    # merge for next phase
    merged_path = os.path.join(cfg["log"]["ckpt_dir"], f"merged_{phase}.pt")
    if hasattr(trainer.model, "merge_and_unload"):
        merged = trainer.model.merge_and_unload()
        torch.save(merged.state_dict(), merged_path)
    return merged_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prune_model', type=str, required=True)
    p.add_argument('--num_epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--micro_batch_size', type=int, default=1)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--lora_r', type=int, default=8)
    p.add_argument('--lora_alpha', type=int, default=16)
    p.add_argument('--lora_dropout', type=float, default=0.05)
    args = p.parse_args()

    # U then V
    ckpt_u = run_phase(args, "U", args.prune_model, "./first_half")
    _      = run_phase(args, "V", ckpt_u, "./second_half")

if __name__ == "__main__":
    main()

    