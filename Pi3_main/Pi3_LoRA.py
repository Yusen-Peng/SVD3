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
from safetensors.torch import load_file, save_file
from accelerate import Accelerator

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)
from trainers.pi3_trainer import Pi3TrainerLoRA
from SVD_LLM.utils.data_utils import *
from SVD_LLM.utils.model_utils import *
from SVD_LLM.evaluater import *
from Pi3_LoRA_helper import build_cfg

device = "cuda" if torch.cuda.is_available() else "cpu"


def run_phase(args, phase, ckpt_in, out_dir):
    cfg = build_cfg(args, phase, ckpt_in, out_dir)
    trainer = Pi3TrainerLoRA(cfg)
    trainer.train()
    # merge for next phase
    merged_path = f"/data/wanghaoxuan/SVD_Pi3_cache/Pi3_lora_{phase}_0.8.safetensors"

    # save the model using accelerate
    accelerator = Accelerator()
    state_dict = accelerator.get_state_dict(trainer.model)
    save_file(state_dict, merged_path)
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

    # lora on U, then on V

    #ckpt_u = run_phase(args, "U", args.prune_model, "./first_half")
    ckpt_u = "/data/wanghaoxuan/SVD_Pi3_cache/Pi3_lora_U_0.8.safetensors"
    _      = run_phase(args, "V", ckpt_u, "./second_half")

if __name__ == "__main__":
    main()

    