import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy
from typing import Dict, Optional
from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin
from .component.svd_pi3 import SVD_Pi3Attention, SVD_Pi3MLP


class TwoFactorLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 W_u: torch.Tensor, W_v: torch.Tensor, bias: Optional[torch.Tensor]):
        super().__init__()
        r = W_v.shape[0]
        assert W_v.shape == (r, in_features)
        assert W_u.shape == (out_features, r)
        self.v = nn.Linear(in_features, r, bias=False)
        self.u = nn.Linear(r, out_features, bias=(bias is not None))
        with torch.no_grad():
            self.v.weight.copy_(W_v)
            self.u.weight.copy_(W_u)
            if bias is not None:
                self.u.bias.copy_(bias)

    def forward(self, x):
        return self.u(self.v(x))


class Pi3(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features        # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)


    def decode(self, hidden, N, H, W):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            hidden = blk(hidden, xpos=pos)

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    
    def forward(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        hidden, pos = self.decode(hidden, N, H, W)

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            conf_hidden = conf_hidden.float()
            conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
        )

# class CompressedPi3(Pi3):
#     """
#     Pi3 variant whose decoder blocks are replaced with SVD-aware Attn/MLP
#     so that factorized checkpoints (…_u/…_v) can be loaded directly.
#     """

#     def __init__(self, pos_type='rope100', decoder_size='large'):
#         super().__init__(pos_type=pos_type, decoder_size=decoder_size)
#         # We keep encoder, camera/point heads, etc. exactly as in Pi3.

#     def _wrap_decoder_with_svd(self):
#         """Replace each decoder block's attn/mlp with SVD-aware modules."""
#         for i, blk in enumerate(self.decoder):
#             assert isinstance(blk, BlockRope)
#             D = blk.attn.qkv.in_features
#             H = getattr(blk.attn, "num_heads", 16)

#             # Make SVD-aware attn/MLP with placeholder ranks (weights set later).
#             svd_attn = SVD_Pi3Attention(
#                 embed_dim=D, num_heads=H,
#                 r_qkv=max(1, D // 4),   # temporary; will be resized by weight load
#                 r_out=max(1, D // 4),
#                 attn_drop_rate=getattr(blk.attn, "attn_drop", 0.0),
#                 proj_drop_rate=getattr(blk.attn, "proj_drop", 0.0),
#                 use_bias_qkv=(blk.attn.qkv.bias is not None),
#                 use_bias_out=(blk.attn.proj.bias is not None),
#                 rope=blk.attn.rope,
#             )
#             self.decoder[i].attn = svd_attn

#             I = blk.mlp.fc1.out_features
#             act_name = blk.mlp.act.__class__.__name__.lower() if hasattr(blk.mlp, "act") else "gelu"
#             svd_mlp = SVD_Pi3MLP(
#                 embed_dim=D, intermediate_dim=I,
#                 r_fc1=max(1, (I + D)//8),
#                 r_fc2=max(1, (I + D)//8),
#                 activation=act_name,
#                 drop_rate=getattr(blk.mlp, "drop", 0.0),
#                 use_bias_fc1=(blk.mlp.fc1.bias is not None),
#                 use_bias_fc2=(blk.mlp.fc2.bias is not None),
#             )
#             self.decoder[i].mlp = svd_mlp

#     @staticmethod
#     def _has_factorized_keys(sd: Dict[str, torch.Tensor]) -> bool:
#         # Quick detector for SVD/whitening-style checkpoints
#         return any(k.endswith("_u.weight") for k in sd.keys()) and any(k.endswith("_v.weight") for k in sd.keys())

#     @torch.no_grad()
#     def load_factorized_state_dict(self, sd: Dict[str, torch.Tensor], strict: bool = True):
#         """
#         Load a factorized (U/V) checkpoint into the SVD-wrapped decoder.
#         Handles attn.o -> attn.proj naming, optional q/k norms, and bias.
#         """
#         if not self._has_factorized_keys(sd):
#             raise ValueError("Provided state_dict does not look factorized (no *_u/*_v keys).")

#         # If the factorized checkpoint used attn 'o_*' names, accept them.
#         def get_attn_proj_name(base: str):
#             # accept either '.attn.o_*' or '.attn.proj_*' in the incoming keys
#             if (base + ".attn.o_u.weight") in sd or (base + ".attn.o_v.weight") in sd:
#                 return "o"
#             return "proj"

#         # Replace decoder with SVD-aware modules
#         self._wrap_decoder_with_svd()

#         # Populate weights block by block
#         for i, blk in enumerate(self.decoder):
#             base = f"decoder.{i}"
#             # ---------- Attention ----------
#             proj_tag = get_attn_proj_name(base)  # "o" or "proj"
#             attn = blk.attn  # SVD_Pi3Attention

#             # qkv ranks/dims
#             qkv_u = sd.get(f"{base}.attn.qkv_u.weight", None)
#             qkv_v = sd.get(f"{base}.attn.qkv_v.weight", None)
#             if qkv_u is not None and qkv_v is not None:
#                 r_qkv = qkv_v.shape[0]
#                 # Rebuild submodules to correct rank
#                 attn.qkv_u = nn.Linear(r_qkv, 3 * attn.embed_dim, bias=("qkv_u.bias" in f"{base}.attn"))
#                 attn.qkv_v = nn.Linear(attn.embed_dim, r_qkv, bias=False)
#                 attn.qkv_u.weight.copy_(qkv_u)
#                 attn.qkv_v.weight.copy_(qkv_v)
#                 qkv_b = sd.get(f"{base}.attn.qkv_u.bias", None)
#                 if qkv_b is not None and attn.qkv_u.bias is not None:
#                     attn.qkv_u.bias.copy_(qkv_b)

#             # out/proj ranks/dims
#             o_u = sd.get(f"{base}.attn.{proj_tag}_u.weight", None)
#             o_v = sd.get(f"{base}.attn.{proj_tag}_v.weight", None)
#             if o_u is not None and o_v is not None:
#                 r_out = o_v.shape[0]
#                 attn.o_u = nn.Linear(r_out, attn.embed_dim, bias=(f"{base}.attn.{proj_tag}_u.bias" in sd))
#                 attn.o_v = nn.Linear(attn.embed_dim, r_out, bias=False)
#                 attn.o_u.weight.copy_(o_u)
#                 attn.o_v.weight.copy_(o_v)
#                 o_b = sd.get(f"{base}.attn.{proj_tag}_u.bias", None)
#                 if o_b is not None and attn.o_u.bias is not None:
#                     attn.o_u.bias.copy_(o_b)

#             # q/k norms (optional in factorized ckpt)
#             for nm in ("q_norm", "k_norm"):
#                 w = sd.get(f"{base}.attn.{nm}.weight", None)
#                 b = sd.get(f"{base}.attn.{nm}.bias", None)
#                 if w is not None and b is not None and hasattr(attn, nm):
#                     getattr(attn, nm).weight.copy_(w)
#                     getattr(attn, nm).bias.copy_(b)

#             # ---------- MLP ----------
#             mlp = blk.mlp  # SVD_Pi3MLP

#             # fc1
#             fc1_u = sd.get(f"{base}.mlp.fc1_u.weight", None)
#             fc1_v = sd.get(f"{base}.mlp.fc1_v.weight", None)
#             if fc1_u is not None and fc1_v is not None:
#                 r1 = fc1_v.shape[0]
#                 out_f = mlp.fc1_u.out_features
#                 in_f  = mlp.fc1_v.in_features
#                 mlp.fc1_u = nn.Linear(r1, out_f, bias=("mlp.fc1_u.bias" in sd))
#                 mlp.fc1_v = nn.Linear(in_f, r1, bias=False)
#                 mlp.fc1_u.weight.copy_(fc1_u)
#                 mlp.fc1_v.weight.copy_(fc1_v)
#                 b = sd.get(f"{base}.mlp.fc1_u.bias", None)
#                 if b is not None and mlp.fc1_u.bias is not None:
#                     mlp.fc1_u.bias.copy_(b)

#             # fc2
#             fc2_u = sd.get(f"{base}.mlp.fc2_u.weight", None)
#             fc2_v = sd.get(f"{base}.mlp.fc2_v.weight", None)
#             if fc2_u is not None and fc2_v is not None:
#                 r2 = fc2_v.shape[0]
#                 out_f = mlp.fc2_u.out_features
#                 in_f  = mlp.fc2_v.in_features
#                 mlp.fc2_u = nn.Linear(r2, out_f, bias=("mlp.fc2_u.bias" in sd))
#                 mlp.fc2_v = nn.Linear(in_f, r2, bias=False)
#                 mlp.fc2_u.weight.copy_(fc2_u)
#                 mlp.fc2_v.weight.copy_(fc2_v)
#                 b = sd.get(f"{base}.mlp.fc2_u.bias", None)
#                 if b is not None and mlp.fc2_u.bias is not None:
#                     mlp.fc2_u.bias.copy_(b)

#         # Load any remaining non-decoder buffers/params (encoder, heads, register_token, etc.)
#         # We ignore mismatch from SVD submodules by filtering keys we already set.
#         leftovers = {}
#         for k, v in sd.items():
#             if (".attn." in k) or (".mlp." in k):
#                 continue
#             leftovers[k] = v
#         # Non-strict load for leftovers to avoid harmless shape diffs
#         _ = super().load_state_dict(leftovers, strict=False)
#         return



class CompressedPi3(Pi3):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace all Linear layers with TwoFactorLinear stubs
        for i, blk in enumerate(self.decoder):
            for sub in ['attn', 'mlp']:
                mod = getattr(blk, sub)
                for name, layer in list(mod.named_children()):
                    if isinstance(layer, nn.Linear):
                        setattr(mod, name,
                                TwoFactorLinear(layer.in_features,
                                                layer.out_features,
                                                W_u=torch.empty(layer.out_features, layer.in_features),
                                                W_v=torch.empty(layer.in_features, layer.in_features),
                                                bias=None))