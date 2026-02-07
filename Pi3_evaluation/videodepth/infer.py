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
from vggt.models.vggt import VGGT
from pi3.models.pi3 import Pi3
from utils.interfaces import infer_videodepth, adaptive_infer_videodepth, learn_entropy_cfg_from_calib, learn_drift_cfg_from_calib, drifting_adaptive_infer_videodepth, learn_augmented_entropy_cfg_from_calib, augmented_adaptive_infer_videodepth
from utils.interfaces import learn_entropy_cfg_continuous_from_calib, fine_grained_adaptive_infer_videodepth
from utils.interfaces import infer_videodepth_VGGT
from utils.files import get_all_sequences, list_imgs_a_sequence
from utils.messages import set_default_arg
from videodepth.utils import save_depth_maps
from utils.interfaces import install_twofactor_modules_from_sd, strip_factor_keys, install_slicabletwofactor_modules_from_sd
from utils.interfaces import vggt_install_twofactor_modules_from_sd

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
    if not USE_VGGT or COMPRESSED:
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

        ADAPTIVE = ('base' in pretrained_model_name_or_path.lower()) and ('baseline' not in pretrained_model_name_or_path.lower())

        if USE_VGGT:
            if not COMPRESSED:
                print(f"🤩🤩🤩Loading the ORIGINAL VGGT from {ckpt}...🤩🤩🤩 on device {device}")
                model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
            else: 
                print(f"🥎🥎🥎Loading the COMPRESSED VGGT from {ckpt}...🥎🥎🥎 on device {device}")
                model = VGGT().to(device).eval()
                vggt_install_twofactor_modules_from_sd(model, sd)
                model.load_state_dict(sd, strict=False)
        else:
            print(f"🥶Loading the ORIGINAL Pi3 from {ckpt}...")
            model = Pi3().to(device).eval()
            model.load_state_dict(sd, strict=True) # enforce it for original Pi3 model
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
            if COMPRESSED and ADAPTIVE:
                if ADAPTIVE_MODE == 'input':
                    if not AUGMENTED:
                        if not FINE_GRAINED:
                            time_used, depth_maps, conf_self = adaptive_infer_videodepth(filelist, model, save_path, hydra_cfg)
                        else:
                            time_used, depth_maps, conf_self = fine_grained_adaptive_infer_videodepth(filelist, model, hydra_cfg)
                    else:
                        time_used, depth_maps, conf_self = augmented_adaptive_infer_videodepth(filelist, model, hydra_cfg)
                elif ADAPTIVE_MODE == 'drift':
                    time_used, depth_maps, conf_self = drifting_adaptive_infer_videodepth(filelist, model, hydra_cfg) 

            else:
                if USE_VGGT:
                    time_used, depth_maps, conf_self = infer_videodepth_VGGT(filelist, model, hydra_cfg)
                else:
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