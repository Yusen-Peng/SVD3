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
from utils.interfaces import infer_monodepth, adaptive_infer_monodepth, embedding_adaptive_infer_monodepth, augmented_adaptive_infer_monodepth, learn_entropy_cfg_from_calib, learn_augmented_entropy_cfg_from_calib, learn_entropy_cfg_from_calib_embedding, learn_drift_cfg_from_calib, drifting_adaptive_infer_monodepth
from utils.files import list_imgs_a_sequence, get_all_sequences
from utils.messages import set_default_arg
from utils.interfaces import install_twofactor_modules_from_sd, strip_factor_keys, install_slicabletwofactor_modules_from_sd

@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    all_eval_datasets: ListConfig      = hydra_cfg.eval_datasets  # see configs/evaluation/monodepth.yaml
    all_data_info: DictConfig          = hydra_cfg.data           # see configs/data/depth.yaml
    pretrained_model_name_or_path: str = hydra_cfg.pi3.pretrained_model_name_or_path  # see configs/evaluation/monodepth.yaml

    # 0. create model
    ADAPTIVE_MODE = 'input' # 'embedding' or 'input' or 'drift' ['input' is the best option so far]
    AUGMENTED = True
    COMPRESSED = True if 'whitening' in pretrained_model_name_or_path.lower() or 'lora' in pretrained_model_name_or_path.lower() or 'baseline' in pretrained_model_name_or_path.lower() else False
    

    device = hydra_cfg.device
    ckpt = pretrained_model_name_or_path
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
            cali_path = "/data/wanghaoxuan/SVD_Pi3_cache/scannet_pi3_calib_nsamples256_size224_seed3.pt"
            cali_white_data = torch.load(cali_path, map_location="cpu")
            if ADAPTIVE_MODE == 'input':
                if not AUGMENTED:
                    print("🍀🍀🍀Learning adaptive entropy cfg from calibration data...🍀🍀🍀")
                    learn_entropy_cfg_from_calib(
                        calib=cali_white_data,
                        save_path='/mnt/extdisk1/wanghaoxuan/SVD-pi3/adaptive_cfg.json',
                        bins=256,
                        tail_frac=0.25,
                        rr_values=(0.1, 0.2, 0.3),
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
        
        # keep track of the average time
        time_dict = []

        
        for seq_idx, seq in enumerate(seq_list):
            # 3.1 list the images in the sequence
            filelist = list_imgs_a_sequence(dataset_info, seq)
            save_dir = osp.join(output_root, seq) if seq is not None else output_root
            os.makedirs(save_dir, exist_ok=True)
            logger.info(f"[{seq_idx}/{len(seq_list)}] Processing {len(filelist)} images to {osp.relpath(save_dir, hydra_cfg.work_dir)}...")

            
            t1 = time.time()
            # 3.2 infer for each image
            for file in tqdm(filelist):
                # 3.2.1 skip if the file already exists
                npy_save_path = osp.join(save_dir, file.split('/')[-1].replace('.png', 'depth.npy'))
                png_save_path = osp.join(save_dir, file.split('/')[-1].replace('.png', 'depth.png'))
                if not hydra_cfg.overwrite and (osp.exists(npy_save_path) and osp.exists(png_save_path)):
                    continue

                # 3.2.2 infer the depth map
                if COMPRESSED and ADAPTIVE:
                    if ADAPTIVE_MODE == 'input':
                        if not AUGMENTED:
                            depth_map = adaptive_infer_monodepth(file, model, hydra_cfg)
                        else:
                            depth_map = augmented_adaptive_infer_monodepth(file, model, hydra_cfg)
                    elif ADAPTIVE_MODE == 'embedding':
                        depth_map = embedding_adaptive_infer_monodepth(file, model, hydra_cfg)
                    elif ADAPTIVE_MODE == 'drift':
                        depth_map = drifting_adaptive_infer_monodepth(file, model, hydra_cfg) 
                else:
                    depth_map = infer_monodepth(file, model, hydra_cfg)

                # 3.2.3 save the depth map to the save_dir as npy
                if isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.cpu().numpy()
                elif not isinstance(depth_map, np.ndarray):
                    raise ValueError(f"Unknown depth map type: {type(depth_map)}")
                np.save(npy_save_path, depth_map)

                # 3.2.4 also save the png
                depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                depth_map = (depth_map * 255).astype(np.uint8)
                cv2.imwrite(png_save_path, depth_map)
            
            t2 = time.time()
            time_dict.append((t2 - t1, len(filelist)))

        # for each dataset
        logger.info(f"Monodepth inference for dataset {dataset_name} finished!")

    del model
    torch.cuda.empty_cache()
    logger.info(f"Monodepth inference for Pi3 finished!")
    
    # dump time_dict into a CSV (headers: time, num_images)
    if len(time_dict) > 0:
        csv_path = "inference_time.csv"
        with open(csv_path, 'w') as f:
            f.write("time,num_images\n")
            for t, n in time_dict:
                f.write(f"{t},{n}\n")
        logger.info(f"Saved inference time to {osp.relpath(csv_path, hydra_cfg.work_dir)}")
        total_time = sum([t for t, n in time_dict])
        total_images = sum([n for t, n in time_dict])
        logger.info(f"Total inference time: {total_time:.2f} seconds for {total_images} images, average time per image: {total_time / total_images:.4f} seconds")

    # compute the throughput based on the csv
    if len(time_dict) > 0:
        total_time = sum([t for t, n in time_dict])
        total_images = sum([n for t, n in time_dict])
        throughput = total_images / total_time
        logger.info(f"Overall throughput: {throughput:.2f} images/second")

if __name__ == "__main__":
    set_default_arg("evaluation", "monodepth")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    main()
