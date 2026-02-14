import hydra
import os
import os.path as osp
import numpy as np
import torch
import time
import torch.nn as nn
from tqdm import tqdm
from omegaconf import DictConfig, ListConfig
from safetensors.torch import load_file

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pi3.models.pi3 import Pi3
from vggt.models.vggt import VGGT
from utils.interfaces import learn_entropy_cfg_from_calib, learn_augmented_entropy_cfg_from_calib, learn_entropy_cfg_from_calib_embedding, learn_drift_cfg_from_calib
from utils.interfaces import learn_entropy_cfg_continuous_from_calib
from utils.interfaces import install_twofactor_modules_from_sd, strip_factor_keys, install_slicabletwofactor_modules_from_sd
from utils.messages import set_default_arg

MiB = 1024 ** 2


def param_bytes(model: Pi3) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())

def buffer_bytes(model: Pi3) -> int:
    return sum(b.numel() * b.element_size() for b in model.buffers())

def count_params(model: Pi3) -> int:
    return sum(p.numel() for p in model.parameters())

def checkpoint_size_mib(path: str) -> float:
    return os.path.getsize(path) / MiB

@torch.inference_mode()
def gpu_param_footprint_mib(model: Pi3) -> float:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    _ = model  # no-op, but keep semantics obvious
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    stable = (param_bytes(model) + buffer_bytes(model)) / MiB
    allocated = after / MiB
    return stable, allocated


@torch.inference_mode()
def benchmark_on_memory(model: Pi3, dataset_cpu: torch.Tensor, autocast_dtype: torch.dtype, warmup: int = 50):
    model.eval()
    device = next(model.parameters()).device

    assert dataset_cpu.ndim == 5, f"Expected (N,1,3,H,W), got {tuple(dataset_cpu.shape)}"
    N = dataset_cpu.shape[0]

    dataset_cpu = dataset_cpu.pin_memory()

    peaks_bytes = np.empty(N, dtype=np.int64)

    # warmup: cycle through first few samples
    for i in range(warmup):
        x = dataset_cpu[i % N : i % N + 1].to(device, non_blocking=True)  # (1,1,3,H,W)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            _ = model(x)
    torch.cuda.synchronize()

    # benchmark
    for i in tqdm(range(N)):
        torch.cuda.reset_peak_memory_stats()
        x = dataset_cpu[i:i+1].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            _ = model(x)
        peaks_bytes[i] = torch.cuda.max_memory_allocated()
    peaks_mib = peaks_bytes / MiB
    return float(peaks_mib.max())



@hydra.main(version_base="1.2", config_path="./configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    all_eval_datasets: ListConfig      = hydra_cfg.eval_datasets  # see configs/evaluation/monodepth.yaml
    all_data_info: DictConfig          = hydra_cfg.data           # see configs/data/depth.yaml
    pretrained_model_name_or_path: str = hydra_cfg.pi3.pretrained_model_name_or_path  # see configs/evaluation/monodepth.yaml

    # 0. create model
    ADAPTIVE_MODE = 'input' # 'embedding' or 'input' or 'drift' ['input' is the best option so far]
    
    # false (not doing augmentation) leads to better results
    AUGMENTED = False
    FINE_GRAINED = False
    DIVERSE_CALI = True


    COMPRESSED = True if 'whitening' in pretrained_model_name_or_path.lower() or 'lora' in pretrained_model_name_or_path.lower() or 'baseline' in pretrained_model_name_or_path.lower() else False
    USE_VGGT = True if 'vggt' in pretrained_model_name_or_path.lower() else False

    device = hydra_cfg.device
    ckpt = pretrained_model_name_or_path
    if not USE_VGGT:
        sd = load_file(ckpt, device=str(device))

    if COMPRESSED:
        print(f"😎Loading the compressed Pi3 from {ckpt}...")
        # Baseline SVD checkpoint saved with .u/.v keys (TwoFactorLinear)
        model = Pi3().to(device).eval()

        ADAPTIVE = True if 'base' in pretrained_model_name_or_path.lower() else False
        if ADAPTIVE:
            # support slicing
            install_slicabletwofactor_modules_from_sd(model, sd)
            sd_rest = strip_factor_keys(sd)
            model.load_state_dict(sd_rest, strict=False)

            # re-load the calibration dataset
            if DIVERSE_CALI:
                cali_path = "/data/wanghaoxuan/yusen_stuff/SVD_Pi3_cache/curation/diverse_pi3_calib_nsamples256_size224_seed3.pt"
                save_path = "/data/wanghaoxuan/yusen_stuff/SVD-pi3/diverse_adaptive_cfg.json"
                print("🌈🌈🌈Using DIVERSE calibration dataset for learning adaptive cfg...🌈🌈🌈")
            else:
                cali_path = "/data/wanghaoxuan/yusen_stuff/SVD_Pi3_cache/curation/scannet_pi3_calib_nsamples256_size224_seed3.pt"
                save_path = "/data/wanghaoxuan/yusen_stuff/SVD-pi3/adaptive_cfg.json"
                print("🌟🌟🌟Using SCANNET calibration dataset for learning adaptive cfg...🌟🌟🌟")

            cali_white_data = torch.load(cali_path, map_location="cpu")
            if ADAPTIVE_MODE == 'input':
                if not AUGMENTED:
                    if not FINE_GRAINED:
                        print("🍀🍀🍀Learning adaptive entropy cfg from calibration data...🍀🍀🍀")
                        learn_entropy_cfg_from_calib(
                            calib=cali_white_data,
                            save_path=save_path,
                            bins=256,
                            tail_frac=0.25,
                            rr_values=(0.1, 0.2, 0.3),
                            device=device
                        )
                    else: 
                        print("🍃🍃🍃Learning adaptive FINE-GRAINED entropy cfg from calibration data...🍃🍃🍃")
                        learn_entropy_cfg_continuous_from_calib(
                            calib=cali_white_data,
                            save_path='/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_finegrained.json',
                            bins=256,
                            rr_min=0.1,
                            rr_max=0.3,
                            rr_target=0.2,
                            alpha=6, # grid search (6, 8, 10) - 
                            device=device
                        )
                else:
                    print("🌟🌟🌟Learning adaptive AUGMENTED entropy cfg from calibration data...🌟🌟🌟")
                    learn_augmented_entropy_cfg_from_calib(
                        calib=cali_white_data,
                        save_path='/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_augmented.json',
                        bins=256,
                        tail_frac=0.25,
                        rr_values=(0.1, 0.2, 0.3),
                        device=device
                    )

            elif ADAPTIVE_MODE == 'embedding':
                print("🩵🩵🩵Learning adaptive entropy cfg from calibration data (embedding)...🩵🩵🩵")
                learn_entropy_cfg_from_calib_embedding(
                    calib=cali_white_data,
                    model=model,
                    save_path='/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_embedding.json',
                    tail_frac=0.25,
                    rr_values=(0.1, 0.2, 0.3),
                    K=64,
                    tau=80.0,
                    soft=False,
                    device=device
                )
            elif ADAPTIVE_MODE == 'drift':
                print("🧨🧨🧨Learning adaptive drifting cfg from calibration data...🧨🧨🧨")
                learn_drift_cfg_from_calib(
                    calib=cali_white_data,
                    model=model,
                    save_path='/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg_drifting.json',
                    tail_frac=0.25,
                    rr_values=(0.1, 0.2, 0.3),
                    device=device
                )

        else:
            install_twofactor_modules_from_sd(model, sd)
            model.load_state_dict(sd, strict=False)
    else:
        if USE_VGGT:
            print(f"🤩🤩🤩Loading the VGGT from {ckpt}...🤩🤩🤩 on device {device}")
            model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        else:
            print(f"🥶🥶🥶Loading the ORIGINAL Pi3 from {ckpt}...🥶🥶🥶")
            model = Pi3().to(device).eval()
            model.load_state_dict(sd, strict=True) # enforce it for original Pi3 model
    model.to(device)

    # synthesize data
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    inputs = torch.randn(256, 1, 3, 224, 224)  # CPU

    # benchmarking
    print("========================================")
    print("checkpoint in GB:", checkpoint_size_mib(ckpt) / 1024)
    print("#params in M", count_params(model) / 1e6)
    peak_mem = benchmark_on_memory(model, inputs, autocast_dtype=torch.float16)
    print(f"Peak GPU memory allocated during inference: {peak_mem:.2f} MiB")
    print("========================================")




if __name__ == "__main__":
    set_default_arg("evaluation", "monodepth")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    main()

