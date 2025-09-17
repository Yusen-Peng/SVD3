import math
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

def drop_p(x):
    # Accept float/int or nn.Dropout or None
    if isinstance(x, (float, int)):
        return float(x)
    if isinstance(x, nn.Dropout):
        return float(x.p)
    return 0.0

class SVD_Pi3Attention(nn.Module):
    """
    A drop-in attention module for Pi3 blocks (replacing FlashAttentionRope).
    qkv/proj are factorized: qkv = U_qkv @ V_qkv, proj = U_o @ V_o
    This uses standard (non-flash) MHA for broad compatibility.

    Args:
        embed_dim: model hidden dim (per token)
        num_heads: attention heads
        r_qkv: low-rank for qkv factorization
        r_out: low-rank for output projection factorization
        attn_drop, proj_drop: dropouts
        use_bias_qkv, use_bias_out: whether U-projections carry bias copied from originals
        rope: optional RoPE2D-like module (if you want to inject RoPE here)
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        r_qkv: int,
        r_out: int,
        attn_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        use_bias_qkv: bool = False,
        use_bias_out: bool = False,
        rope=None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_v = nn.Linear(embed_dim, r_qkv, bias=False)
        self.qkv_u = nn.Linear(r_qkv, 3 * embed_dim, bias=use_bias_qkv)

        self.o_v = nn.Linear(embed_dim, r_out, bias=False)
        self.o_u = nn.Linear(r_out, embed_dim, bias=use_bias_out)

        p_attn = drop_p(attn_drop_rate)
        p_proj = drop_p(proj_drop_rate)
        self.attn_drop = nn.Dropout(p_attn) if p_attn > 0.0 else nn.Identity()
        self.proj_drop = nn.Dropout(p_proj) if p_proj > 0.0 else nn.Identity()
        self.rope = rope

    def _split_heads(self, t: Tensor, B: int, T: int) -> Tensor:
        return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _merge_heads(self, t: Tensor, B: int, T: int) -> Tensor:
        return t.transpose(1, 2).reshape(B, T, self.embed_dim)

    def forward(
        self,
        x: Tensor,                     # (B, T, C)
        xpos: Optional[Tensor] = None, # (B, T, 2) if using 2D RoPE, else None
        attention_mask: Optional[Tensor] = None,  # (B, 1, T, S) if provided
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs
    ):
        B, T, C = x.shape

        # Low-rank qkv
        qkv = self.qkv_u(self.qkv_v(x))              # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)               # each (B, T, C)

        q = self._split_heads(q, B, T)               # (B, H, T, Hd)
        k = self._split_heads(k, B, T)
        v = self._split_heads(v, B, T)

        # Optional: apply RoPE if your pipeline expects attn module to handle it
        # If BlockRope already handles RoPE externally, you can skip this.
        if self.rope is not None and xpos is not None:
            # Typical pattern: self.rope expects (q, k, xpos) and returns rotated (q, k)
            # Adjust this to your exact RoPE2D API if needed.
            try:
                q, k = self.rope(q, k, xpos=xpos)
            except Exception:
                # If your rope API differs, leave as-is to keep it robust
                pass

        # Append cache
        kv_len = k.shape[-2]
        if past_key_value is not None:
            pk, pv = past_key_value
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
            kv_len = k.shape[-2]

        new_past = (k, v) if use_cache else None

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,T,S)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
            min_val = torch.finfo(attn_scores.dtype).min
            attn_scores = torch.max(attn_scores, torch.tensor(min_val, device=attn_scores.device))

        attn = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn = self.attn_drop(attn)

        y = torch.matmul(attn, v)                    # (B, H, T, Hd)
        y = self._merge_heads(y, B, T)               # (B, T, C)

        # Low-rank output projection
        y = self.o_u(self.o_v(y))
        y = self.proj_drop(y)

        if not output_attentions:
            attn = None
        #return y, attn, new_past
        return y

class SVD_Pi3MLP(nn.Module):
    """
    Factorized MLP matching DINOv2 Mlp(fc1 -> act -> fc2) shape/behavior.
    fc1 = U1 @ V1, fc2 = U2 @ V2
    """
    def __init__(
        self,
        embed_dim: int,
        intermediate_dim: int,
        r_fc1: int,
        r_fc2: int,
        activation: str = "gelu",
        drop_rate: float = 0.0,
        use_bias_fc1: bool = False,
        use_bias_fc2: bool = False,
    ):
        super().__init__()
        self.fc1_v = nn.Linear(embed_dim, r_fc1, bias=False)
        self.fc1_u = nn.Linear(r_fc1, intermediate_dim, bias=use_bias_fc1)

        self.fc2_v = nn.Linear(intermediate_dim, r_fc2, bias=False)
        self.fc2_u = nn.Linear(r_fc2, embed_dim, bias=use_bias_fc2)

        if activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.GELU()

        p_drop = drop_p(drop_rate)
        self.drop = nn.Dropout(p_drop) if p_drop > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1_u(self.fc1_v(x))
        x = self.act(x)
        x = self.fc2_u(self.fc2_v(x))
        x = self.drop(x)
        return x

