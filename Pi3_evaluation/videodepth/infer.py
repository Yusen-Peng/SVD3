import hydra
import os
import os.path as osp
import torch
import logging
import json
from omegaconf import DictConfig, ListConfig
from safetensors.torch import load_file
import rootutils
import torch.nn as nn
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pi3.models.pi3 import Pi3
from utils.interfaces import infer_videodepth
from utils.files import get_all_sequences, list_imgs_a_sequence
from utils.messages import set_default_arg
from videodepth.utils import save_depth_maps

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
                old = getattr(parent, leaf_name)  # original nn.Linear

                in_f, out_f = old.in_features, old.out_features
                r = sd[k_v_w].shape[0]
                has_bias = (k_u_b in sd)

                # Build TwoFactorLinear with correct geometry
                tfl = TwoFactorLinear(in_features=in_f, out_features=out_f, r=r, has_bias=has_bias)
                tfl = tfl.to(device=old.weight.device, dtype=old.weight.dtype)

                setattr(parent, leaf_name, tfl)
    return model


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    all_eval_datasets: ListConfig      = hydra_cfg.eval_datasets  # see configs/evaluation/videodepth.yaml
    all_data_info: DictConfig          = hydra_cfg.data           # see configs/data/depth.yaml
    pretrained_model_name_or_path: str = hydra_cfg.pi3.pretrained_model_name_or_path  # see configs/evaluation/videodepth.yaml

    # 0. create model
    # model = Pi3.from_pretrained(pretrained_model_name_or_path).to(hydra_cfg.device).eval()

    # 0. create model
    COMPRESSED = True if 'whitening' in pretrained_model_name_or_path.lower() or 'lora' in pretrained_model_name_or_path.lower() or 'baseline' in pretrained_model_name_or_path.lower() else False
    device = hydra_cfg.device
    ckpt = pretrained_model_name_or_path
    sd = load_file(ckpt, device=str(device))
    if COMPRESSED:
        print(f"😎Loading the compressed Pi3 from {ckpt}...")
        # Baseline SVD checkpoint saved with .u/.v keys (TwoFactorLinear)
        model = Pi3().to(device).eval()
        install_twofactor_modules_from_sd(model, sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            print("Note: unexpected keys (benign):", unexpected)
        if missing:
            print("Note: missing keys (benign if non-decoder):", missing)
    else:
        print(f"🥶Loading the ORIGINAL Pi3 from {ckpt}...")
        model = Pi3().to(device).eval()
        model.load_state_dict(sd)
    model.to(device)

    logger = logging.getLogger("videodepth-infer")
    logger.info(f"Loaded Pi3 from {pretrained_model_name_or_path}")

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1. look up dataset config from configs/data
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
        dataset_info = all_data_info[dataset_name]

        # 2. get the sequence list
        if dataset_info.type == "video":
            # most of the datasets have many sequences of video
            seq_list = get_all_sequences(dataset_info)
        elif dataset_info.type == "mono":
            raise ValueError("dataset type `mono` is not supported for videodepth evaluation")
        else:
            raise ValueError(f"Unknown dataset type: {dataset_info.type}")

        model = model.eval()
        output_root = osp.join(hydra_cfg.output_dir, dataset_name)
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Infering videodepth on {dataset_name} dataset..., output to {osp.relpath(output_root, hydra_cfg.work_dir)}")

        # 3. infer for each sequence (video)
        for seq_idx, seq in enumerate(seq_list, start=1):
            filelist = list_imgs_a_sequence(dataset_info, seq)
            save_dir = osp.join(output_root, seq)

            if not hydra_cfg.overwrite and (osp.isdir(save_dir) and len(os.listdir(save_dir)) == 2 * len(filelist) + 1):
                logger.info(f"[{seq_idx}/{len(seq_list)}] Sequence {seq} already processed, skipping.")
                continue
            
            # time_used: float, or List[float] (len = 2)
            # depth_maps: (N, H, W), torch.Tensor
            # conf_self: (N, H, W) torch.Tensor, or just None is ok
            time_used, depth_maps, conf_self = infer_videodepth(filelist, model, hydra_cfg)
            logger.info(f"[{seq_idx}/{len(seq_list)}] Sequence {seq} processed, time: {time_used}, saving depth maps...")

            os.makedirs(save_dir, exist_ok=True)
            save_depth_maps(depth_maps, save_dir, conf_self=conf_self)
            # save time
            with open(osp.join(save_dir, "_time.json"), "w") as f:
                json.dump({
                    "time": time_used,
                    "frames": len(filelist),
                }, f, indent=4)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    set_default_arg("evaluation", "videodepth")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()