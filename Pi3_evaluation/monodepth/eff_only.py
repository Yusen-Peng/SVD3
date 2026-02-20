import hydra
import os
import os.path as osp
import numpy as np
import cv2
import logging
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
from utils.interfaces import adaptive_infer_monodepth_efficiency_only
from utils.interfaces import infer_monodepth, adaptive_infer_monodepth, embedding_adaptive_infer_monodepth, augmented_adaptive_infer_monodepth, learn_entropy_cfg_from_calib, learn_augmented_entropy_cfg_from_calib, learn_entropy_cfg_from_calib_embedding, learn_drift_cfg_from_calib, drifting_adaptive_infer_monodepth
from utils.interfaces import learn_entropy_cfg_continuous_from_calib, fine_grained_adaptive_infer_monodepth
from utils.interfaces import infer_monodepth_VGGT
from utils.files import list_imgs_a_sequence, get_all_sequences
from utils.messages import set_default_arg
from utils.interfaces import install_twofactor_modules_from_sd, strip_factor_keys, install_slicabletwofactor_modules_from_sd, vggt_install_twofactor_modules_from_sd, vggt_install_slicabletwofactor_modules_from_sd
from utils.interfaces import adaptive_infer_monodepth_VGGT
from utils.constants import BASE_RR, PI3_05_GFLOP, PI3_04_GFLOP, PI3_03_GFLOP, PI3_02_GFLOP, PI3_01_GFLOP, VGGT_05_GFLOP, VGGT_04_GFLOP, VGGT_03_GFLOP, VGGT_02_GFLOP, VGGT_01_GFLOP


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    all_eval_datasets: ListConfig      = hydra_cfg.eval_datasets  # see configs/evaluation/monodepth.yaml
    all_data_info: DictConfig          = hydra_cfg.data           # see configs/data/depth.yaml
    pretrained_model_name_or_path: str = hydra_cfg.pi3.pretrained_model_name_or_path  # see configs/evaluation/monodepth.yaml

    # 0. create model
    ADAPTIVE_MODE = 'input' # 'embedding' or 'input' or 'drift' ['input' is the best option so far]
    AUGMENTED = False
    FINE_GRAINED = False
    DIVERSE_CALI = False

    COMPRESSED = True if 'whitening' in pretrained_model_name_or_path.lower() or 'lora' in pretrained_model_name_or_path.lower() or 'baseline' in pretrained_model_name_or_path.lower() else False
    USE_VGGT = True if 'vggt' in pretrained_model_name_or_path.lower() else False

    device = hydra_cfg.device
    ckpt = pretrained_model_name_or_path
    sd = load_file(ckpt, device=str(device))

    if COMPRESSED and not USE_VGGT:
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

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1. look up dataset config from configs/data
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        dataset_info = all_data_info[dataset_name]

        # 2. get the sequence list
        if dataset_info.type == "video":
            # most of the datasets have many sequences of video
            seq_list = get_all_sequences(dataset_info)
        elif dataset_info.type == "mono":
            # some datasets (like nyu-v2) have only a set of images, only for monodepth
            seq_list = [None]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_info.type}")

        # 3. infer for each sequence
        output_root = osp.join(hydra_cfg.output_dir, dataset_name)
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Infering monodepth on {dataset_name} dataset..., output to {osp.relpath(output_root, hydra_cfg.work_dir)}")
        

        regimes_count = {"low": 0, "medium": 0, "high": 0}
        
        
        for seq_idx, seq in enumerate(seq_list):
            # 3.1 list the images in the sequence
            filelist = list_imgs_a_sequence(dataset_info, seq)
            save_dir = osp.join(output_root, seq) if seq is not None else output_root
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"[{seq_idx}/{len(seq_list)}] Processing {len(filelist)} images to {osp.relpath(save_dir, hydra_cfg.work_dir)}...")

            # 3.2 infer for each image
            for file in tqdm(filelist):
                # 3.2.1 skip if the file already exists
                npy_save_path = osp.join(save_dir, file.split('/')[-1].replace('.png', 'depth.npy'))
                png_save_path = osp.join(save_dir, file.split('/')[-1].replace('.png', 'depth.png'))
                if not hydra_cfg.overwrite and (osp.exists(npy_save_path) and osp.exists(png_save_path)):
                    continue

                regime, _ = adaptive_infer_monodepth_efficiency_only(file, model, save_path, hydra_cfg)
                regimes_count[regime] += 1
        

        average_GFLOPs = 0.0
        if BASE_RR == 0.4:
            average_GFLOPs = (regimes_count["low"] * PI3_01_GFLOP + regimes_count["medium"] * PI3_02_GFLOP + regimes_count["high"] * PI3_03_GFLOP) / sum(regimes_count.values())
        elif BASE_RR == 0.5:
            average_GFLOPs = (regimes_count["low"] * PI3_02_GFLOP + regimes_count["medium"] * PI3_03_GFLOP + regimes_count["high"] * PI3_04_GFLOP) / sum(regimes_count.values())
        elif BASE_RR == 0.6:
            average_GFLOPs = (regimes_count["low"] * PI3_03_GFLOP + regimes_count["medium"] * PI3_04_GFLOP + regimes_count["high"] * PI3_05_GFLOP) / sum(regimes_count.values())
        else:
            raise ValueError(f"Unknown BASE_RR value: {BASE_RR}")

        # report
        print("================================")
        print(f"Dataset: {dataset_name}")
        print(f"Regime distribution: {regimes_count}")
        target = None
        if BASE_RR == 0.4:
            target = "compression 80%"
        elif BASE_RR == 0.5:
            target = "compression 70%"
        elif BASE_RR == 0.6:
            target = "compression 60%"
        else:
            raise ValueError(f"Unknown BASE_RR value: {BASE_RR}")

        print(f"compression target: {target}")
        print(f"Average GFLOPs: {average_GFLOPs:.2f}")
        print("================================")




            

    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    set_default_arg("evaluation", "monodepth")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    main()
