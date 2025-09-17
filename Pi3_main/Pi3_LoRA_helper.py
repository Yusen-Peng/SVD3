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
