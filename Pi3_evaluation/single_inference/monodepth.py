import cv2
import hydra
import os
import os.path as osp
import torch
import logging
import json
from omegaconf import DictConfig, ListConfig
from safetensors.torch import load_file
import rootutils
import numpy as np
import torch.nn as nn
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from vggt.models.vggt import VGGT
from pi3.models.pi3 import Pi3
from utils.interfaces import infer_monodepth, adaptive_infer_monodepth, embedding_adaptive_infer_monodepth, augmented_adaptive_infer_monodepth, learn_entropy_cfg_from_calib, learn_augmented_entropy_cfg_from_calib, learn_entropy_cfg_from_calib_embedding, learn_drift_cfg_from_calib, drifting_adaptive_infer_monodepth
from utils.interfaces import learn_entropy_cfg_continuous_from_calib, fine_grained_adaptive_infer_monodepth
from utils.interfaces import infer_monodepth_VGGT
from utils.files import list_imgs_a_sequence, get_all_sequences
from utils.messages import set_default_arg
from utils.interfaces import install_twofactor_modules_from_sd, strip_factor_keys, install_slicabletwofactor_modules_from_sd, vggt_install_twofactor_modules_from_sd, vggt_install_slicabletwofactor_modules_from_sd
from utils.interfaces import adaptive_infer_monodepth_VGGT
from utils.depth import depth_evaluation, EVAL_DEPTH_METADATA


import numpy as np
import matplotlib.pyplot as plt

def save_depth_matplotlib_png(depth_map: np.ndarray, png_save_path: str,
                              cmap="magma", vmin=None, vmax=None):
    d = depth_map.astype(np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    if vmin is None or vmax is None:
        vmin = np.percentile(d, 1)
        vmax = np.percentile(d, 99)

    plt.imsave(png_save_path, d, cmap=cmap, vmin=vmin, vmax=vmax)

def save_depth_colormap_png(depth_map: np.ndarray, png_save_path: str,
                            vmin=None, vmax=None, use_percentile=True):
    d = depth_map.astype(np.float32)

    # handle bad values
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    # choose normalization range
    if vmin is None or vmax is None:
        if use_percentile:
            vmin = np.percentile(d, 1)
            vmax = np.percentile(d, 99)
        else:
            vmin, vmax = float(d.min()), float(d.max())

    denom = max(vmax - vmin, 1e-6)
    d_norm = np.clip((d - vmin) / denom, 0.0, 1.0)

    # to 8-bit
    d_u8 = (d_norm * 255).astype(np.uint8)

    # apply colormap (try: INFERNO / MAGMA / JET / TURBO / VIRIDIS)
    colored = cv2.applyColorMap(d_u8, cv2.COLORMAP_INFERNO)

    cv2.imwrite(png_save_path, colored)



@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    
    
    model_path = "/data/wanghaoxuan/yusen_stuff/SVD_Pi3_cache/Pi3_svd_baseline_0.4_BASE.safetensors"
    #model_path = "/data/wanghaoxuan/yusen_stuff/SVD_Pi3_cache/Pi3_svd_baseline_0.2.safetensors"

    image_file = "/data/wanghaoxuan/yusen_stuff/SVD-pi3/Pi3_evaluation/single_inference/sample_images/high_sample.jpg"
    gt_path = "/data/wanghaoxuan/yusen_stuff/SVD-pi3/Pi3_evaluation/single_inference/sample_images/high_depth.png"

    # image_file = "/data/wanghaoxuan/yusen_stuff/SVD-pi3/Pi3_evaluation/single_inference/sample_images/mid_sample.jpg"
    # gt_path = "/data/wanghaoxuan/yusen_stuff/SVD-pi3/Pi3_evaluation/single_inference/sample_images/mid_depth.png"

    
    
    
    pretrained_model_name_or_path = model_path
    save_dir = "/data/wanghaoxuan/yusen_stuff/SVD-pi3/Pi3_evaluation/single_inference/saved_depths"


    ADAPTIVE_MODE = 'input' # 'embedding' or 'input' or 'drift' ['input' is the best option so far]
    AUGMENTED = False
    FINE_GRAINED = False
    DIVERSE_CALI = False

    COMPRESSED = True if 'whitening' in pretrained_model_name_or_path.lower() or 'lora' in pretrained_model_name_or_path.lower() or 'baseline' in pretrained_model_name_or_path.lower() else False
    USE_VGGT = True if 'vggt' in pretrained_model_name_or_path.lower() else False

    device = 'cuda'
    ckpt = pretrained_model_name_or_path
    sd = load_file(ckpt, device=str(device))

    if COMPRESSED and not USE_VGGT:
        print(f"😎Loading the compressed Pi3 from {ckpt}...")
        # Baseline SVD checkpoint saved with .u/.v keys (TwoFactorLinear)
        model = Pi3().to(device).eval()

        ADAPTIVE = True if 'BASE' in pretrained_model_name_or_path else False
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
                cali_path = "/data/wanghaoxuan/yusen_stuff/SVD_Pi3_cache/scannet_pi3_calib_nsamples256_size224_seed3.pt"
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

        ADAPTIVE = 'BASE' in pretrained_model_name_or_path

        if USE_VGGT:
            if not COMPRESSED:
                print(f"🤩🤩🤩Loading the ORIGINAL VGGT from {ckpt}...🤩🤩🤩 on device {device}")
                model = VGGT().to(device).eval()
                model.load_state_dict(sd, strict=True)
            else:
                if not ADAPTIVE:
                    print(f"🥎🥎🥎Loading the COMPRESSED VGGT from {ckpt}...🥎🥎🥎 on device {device}")
                    model = VGGT().to(device).eval()
                    vggt_install_twofactor_modules_from_sd(model, sd)
                    model.load_state_dict(sd, strict=False)
                else:
                    print(f"🏈🏈🏈Loading the COMPRESSED and ADAPTIVE VGGT from {ckpt}...🏈🏈🏈 on device {device}")
                    save_path = "/data/wanghaoxuan/yusen_stuff/SVD-pi3/adaptive_cfg.json"
                    
                    cali_path = "/data/wanghaoxuan/yusen_stuff/SVD_Pi3_cache/scannet_pi3_calib_nsamples256_size224_seed3.pt"
                    cali_white_data = torch.load(cali_path, map_location="cpu")

                    learn_entropy_cfg_from_calib(
                        calib=cali_white_data,
                        save_path=save_path,
                        bins=256,
                        tail_frac=0.25,
                        rr_values=(0.1, 0.2, 0.3),
                        device=device
                    )
                    
                    model = VGGT().to(device).eval()
                    vggt_install_slicabletwofactor_modules_from_sd(model, sd)
                    model.load_state_dict(sd, strict=False)
        else:
            print(f"🥶Loading the ORIGINAL Pi3 from {ckpt}...")
            model = Pi3().to(device).eval()
            model.load_state_dict(sd, strict=True) # enforce it for original Pi3 model
    model.to(device)

    logger = logging.getLogger("monodepth-infer")
    logger.info(f"Loaded Pi3 from {pretrained_model_name_or_path}")

    file = image_file
    png_save_path = osp.join(save_dir, file.split('/')[-1].replace('.jpg', 'depth.png'))
    # 3.2.3 save the depth map to the save_dir as npy
    npy_save_path = osp.join(save_dir, file.split('/')[-1].replace('.jpg', 'depth.npy'))


    if COMPRESSED and ADAPTIVE and not USE_VGGT:
        if ADAPTIVE_MODE == 'input':
            if not AUGMENTED:
                if not FINE_GRAINED:
                    depth_map = adaptive_infer_monodepth(file, model, save_path, hydra_cfg, verbose=True)
                else:
                    depth_map = fine_grained_adaptive_infer_monodepth(file, model, hydra_cfg)
            else:
                depth_map = augmented_adaptive_infer_monodepth(file, model, hydra_cfg)
        elif ADAPTIVE_MODE == 'drift':
            depth_map = drifting_adaptive_infer_monodepth(file, model, hydra_cfg) 
    else:
        if USE_VGGT:
            if not ADAPTIVE:
                depth_map = infer_monodepth_VGGT(file, model, hydra_cfg)
            else:
                depth_map = adaptive_infer_monodepth_VGGT(file, model, save_path, hydra_cfg)
        else:
            depth_map = infer_monodepth(file, model, hydra_cfg)


    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    elif not isinstance(depth_map, np.ndarray):
        raise ValueError(f"Unknown depth map type: {type(depth_map)}")
    np.save(npy_save_path, depth_map)


    # 3.2.4 also save the png
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map = (depth_map * 255).astype(np.uint8)
    cv2.imwrite(png_save_path, depth_map)



    del model
    pred_depth = np.load(npy_save_path)
    
    save_depth_matplotlib_png(pred_depth, png_save_path.replace(".png", "_matplotlib.png"))
    save_depth_colormap_png(pred_depth, png_save_path.replace(".png", "_inferno.png"))
    
    
    mono_metadata = EVAL_DEPTH_METADATA.get("scannet", None)
    depth_read_func = mono_metadata["depth_read_func"]
    gt_depth = depth_read_func(gt_path)

    Hgt, Wgt = gt_depth.shape[:2]
    Hp, Wp = pred_depth.shape[:2]

    if (Hp, Wp) != (Hgt, Wgt):
        # resize prediction to GT size (recommended)
        pred_depth = cv2.resize(pred_depth.astype(np.float32), (Wgt, Hgt), interpolation=cv2.INTER_LINEAR)

    depth_results, _, _, _ = depth_evaluation(pred_depth, gt_depth)
    print(depth_results)




if __name__ == "__main__":
    set_default_arg("evaluation", "videodepth")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()
