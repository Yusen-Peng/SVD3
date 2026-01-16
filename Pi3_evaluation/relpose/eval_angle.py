
import os
import os.path as osp
import numpy as np
import torch
import hydra
import logging
import json

from omegaconf import DictConfig, ListConfig
from tqdm import tqdm
from safetensors.torch import load_file

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pi3.models.pi3 import Pi3
from utils.interfaces import infer_cameras_w2c, learn_entropy_cfg_from_calib
from utils.messages import set_default_arg, write_csv
from relpose.metric import se3_to_relative_pose_error, calculate_auc_np
from utils.interfaces import install_twofactor_modules_from_sd, strip_factor_keys, install_slicabletwofactor_modules_from_sd

@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    all_eval_datasets: ListConfig      = hydra_cfg.eval_datasets  # see configs/evaluation/monodepth.yaml
    all_data_info: DictConfig          = hydra_cfg.data           # see configs/data/depth.yaml
    pretrained_model_name_or_path: str = hydra_cfg.pi3.pretrained_model_name_or_path  # see configs/evaluation/monodepth.yaml

    # 0. create model
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
            install_twofactor_modules_from_sd(model, sd)
            model.load_state_dict(sd, strict=False)
    else:
        print(f"🥶Loading the ORIGINAL Pi3 from {ckpt}...")
        model = Pi3().to(device).eval()
        model.load_state_dict(sd, strict=True) # enforce it for original Pi3 model
    model.to(device)

    logger = logging.getLogger("relpose-angle")
    logger.info(f"Loaded Pi3 from {pretrained_model_name_or_path}")
    
    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1. look up dataset config from configs/data, decide the dataset name
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
        dataset_info = all_data_info[dataset_name]
        dataset = hydra.utils.instantiate(dataset_info.cfg)

        # 2. ready to read, and look up sampled ids from sequence name
        model.eval()
        sample_config: DictConfig = dataset_info.sampling
        logger.info(f"Sampling strategy: {sample_config.strategy}")
        with open(dataset_info.seq_id_map, "r") as f:
            seq_id_map = json.load(f)

        # 3. prepare for metrics
        rError = []
        tError = []
        metric_dict: dict = {}
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Evaluating {dataset_name} with Pi3...")
        tbar = tqdm(dataset.sequence_list, desc=f"[{dataset_name} eval]")
        for seq_name in tbar:
            # 4. decide sampling strategy to choose sample frames, from all frames (seq_num_frames) of a sequence
            ids = seq_id_map[seq_name]

            # 5. load data sample (only extrinsics are used)
            batch = dataset.get_data(sequence_name=seq_name, ids=ids)
            gt_extrs = batch["extrs"]
            
            with torch.amp.autocast(device_type=hydra_cfg.device, dtype=torch.float64):
                # 6. infer cameras
                pred_extrs, pred_intrs = infer_cameras_w2c(batch['image_paths'], model, hydra_cfg)

                # 7. compute metrics
                rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
                    pred_se3   = pred_extrs,
                    gt_se3     = gt_extrs,
                    # num_frames = num_frames,
                    num_frames = len(ids),
                )

            # 8. update metric for a sequence
            tbar.set_postfix_str(f"Sequence {seq_name} RotErr(Deg): {rel_rangle_deg.mean():5.2f} | TransErr(Deg): {rel_tangle_deg.mean():5.2f}")
            # logger.info(f"Sequence {seq_name} RotErr(Deg): {rel_rangle_deg.mean():5.2f} | TransErr(Deg): {rel_tangle_deg.mean():5.2f}")

            rError.extend(rel_rangle_deg.cpu().numpy())
            tError.extend(rel_tangle_deg.cpu().numpy())
        
        rError = np.array(rError)
        tError = np.array(tError)
        # 9. arrange all intermediate results to metrics
        for threshold in dataset_info.metric_thresholds:
            metric_dict[f"Racc_{threshold}"] = np.mean(rError < threshold).item() * 100
            metric_dict[f"Tacc_{threshold}"] = np.mean(tError < threshold).item() * 100
            Auc, _ = calculate_auc_np(rError, tError, max_threshold=threshold)
            metric_dict[f"Auc_{threshold}"]  = Auc.item() * 100

        logger.info(f"{dataset_name} - Average pose estimation metrics: {metric_dict}")

        # 9. save evaluation metrics to csv
        statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric")  # + ".csv"
        if getattr(hydra_cfg, "save_suffix", None) is not None:
            statistics_file += f"-{hydra_cfg.save_suffix}"
        statistics_file += ".csv"
        write_csv(statistics_file, metric_dict)

    del model
    torch.cuda.empty_cache()
    logger.info(f"Finished evaluating model Pi3 on all datasets.")


if __name__ == "__main__":
    set_default_arg("evaluation", "relpose-angular")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()
