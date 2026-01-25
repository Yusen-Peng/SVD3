import os
import os.path as osp
import logging
import numpy as np
import torch
import hydra
import torch.nn as nn
from tqdm import tqdm
import omegaconf
from omegaconf import DictConfig, ListConfig
from safetensors.torch import load_file

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pi3.models.pi3 import Pi3
from vggt.models.vggt import VGGT
from utils.interfaces import infer_cameras_c2w, adaptive_infer_cameras_c2w, learn_entropy_cfg_from_calib, learn_drift_cfg_from_calib, drifting_adaptive_infer_cameras_c2w, learn_augmented_entropy_cfg_from_calib, augmented_adaptive_infer_cameras_c2w
from utils.interfaces import learn_entropy_cfg_continuous_from_calib, fine_grained_adaptive_infer_cameras_c2w
from utils.interfaces import infer_cameras_c2w_VGGT
from utils.files import list_imgs_a_sequence, get_all_sequences
from utils.messages import set_default_arg, write_csv, save_list_of_matrices
from relpose.evo_utils import calculate_averages, load_traj, eval_metrics, plot_trajectory, get_tum_poses, save_tum_poses
from utils.interfaces import install_twofactor_modules_from_sd, strip_factor_keys, install_slicabletwofactor_modules_from_sd

@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    all_eval_datasets: ListConfig      = hydra_cfg.eval_datasets  # see configs/evaluation/monodepth.yaml
    all_data_info: DictConfig          = hydra_cfg.data           # see configs/data/depth.yaml
    pretrained_model_name_or_path: str = hydra_cfg.pi3.pretrained_model_name_or_path  # see configs/evaluation/monodepth.yaml

    # 0. create model
    ADAPTIVE_MODE = 'input' # 'embedding' or 'input' or 'drift' ['input' is the best option so far]
    AUGMENTED = False
    FINE_GRAINED = True
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
            cali_path = "/data/wanghaoxuan/SVD_Pi3_cache/scannet_pi3_calib_nsamples256_size224_seed3.pt"
            cali_white_data = torch.load(cali_path, map_location="cpu")
            if ADAPTIVE_MODE == 'input':
                if not AUGMENTED:
                    if not FINE_GRAINED:
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
        if USE_VGGT:
            print(f"🤩🤩🤩Loading the VGGT from {ckpt}...🤩🤩🤩 on device {device}")
            model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
        else:
            print(f"🥶Loading the ORIGINAL Pi3 from {ckpt}...")
            model = Pi3().to(device).eval()
            model.load_state_dict(sd, strict=True) # enforce it for original Pi3 model
    model.to(device)


    logger = logging.getLogger(f"relpose-dist")
    logger.info(f"Loaded Pi3 from {pretrained_model_name_or_path}")

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1. look up dataset config from configs/data, decide the dataset name
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        dataset_info = all_data_info[dataset_name]

        # 2. get the sequence list
        seq_list = get_all_sequences(dataset_info)
        output_root = osp.join(hydra_cfg.output_dir, dataset_name)
        os.makedirs(output_root, exist_ok=True)

        # 3. infer for each sequence
        model = model.eval()
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Infering relpose(c2w) on {dataset_name} dataset..., output to {osp.relpath(output_root, hydra_cfg.work_dir)}")

        results = []
        tbar = tqdm(seq_list, desc=f"[{dataset_name} eval]")
        for seq in tbar:
            # 4.1 list all images of this sequence
            filelist = list_imgs_a_sequence(dataset_info, seq)
            filelist = filelist[:: hydra_cfg.pose_eval_stride]

            # 4.2 real inference
            # pr_poses: c2w poses, (N, 3, 4), in torch
            # pr_intrs: focals + pps, (N, 3, 3), in numpy
            if COMPRESSED and ADAPTIVE:
                if ADAPTIVE_MODE == 'input':
                    if not AUGMENTED:
                        if not FINE_GRAINED:
                            pr_poses, pr_intrs = adaptive_infer_cameras_c2w(filelist, model, hydra_cfg)
                        else:
                            pr_poses, pr_intrs = fine_grained_adaptive_infer_cameras_c2w(filelist, model, hydra_cfg)
                    else:
                        pr_poses, pr_intrs = augmented_adaptive_infer_cameras_c2w(filelist, model, hydra_cfg)
                elif ADAPTIVE_MODE == 'drift':
                    pr_poses, pr_intrs = drifting_adaptive_infer_cameras_c2w(filelist, model, hydra_cfg)
            else:
                if USE_VGGT:
                    pr_poses, pr_intrs = infer_cameras_c2w_VGGT(filelist, model, hydra_cfg)
                else:
                    pr_poses, pr_intrs = infer_cameras_c2w(filelist, model, hydra_cfg)
            pred_traj = get_tum_poses(pr_poses)

            # 4.3 save predicted poses & intrinsics
            seq_save_dir = osp.join(output_root, seq)
            os.makedirs(seq_save_dir, exist_ok=True)
            # save predicted poses
            save_tum_poses(pred_traj, osp.join(output_root, seq, "pred_traj.txt"), verbose=hydra_cfg.verbose)
            np.save(osp.join(seq_save_dir, "pred_poses.npy"), pr_poses)
            save_list_of_matrices(pr_poses.numpy().tolist(), osp.join(seq_save_dir, "pred_intrinsics.json"))
            # save predicted intrinsics (if available)
            if pr_intrs is not None:
                np.save(osp.join(seq_save_dir, "pred_intrinsics.npy"), pr_intrs)
                save_list_of_matrices(pr_intrs.tolist(), osp.join(seq_save_dir, "pred_intrinsics.json"))

            # 4.4 read ground truth trajectory
            try:
                gt_traj = load_traj(
                    gt_traj_file = dataset_info.anno.path.format(seq=seq),
                    traj_format  = dataset_info.anno.format,
                    stride       = hydra_cfg.pose_eval_stride,
                )
            except np.linalg.LinAlgError:
                logger.warning(f"Failed to load ground truth trajectory for sequence {seq} in dataset {dataset_name}.")
                continue

            # 4.5 evaluate predicted trajectory with ground truth trajectory, plot the trajectory
            if gt_traj is not None:
                ate, rpe_trans, rpe_rot = eval_metrics(
                    pred_traj, gt_traj,
                    seq      = seq,
                    filename = osp.join(output_root, seq, "eval_metric.txt"),
                    verbose  = hydra_cfg.verbose,
                )
                plot_trajectory(pred_traj, gt_traj, title=seq, filename=osp.join(output_root, seq, "vis.png"), verbose=hydra_cfg.verbose)
            else:
                raise ValueError(f"Ground truth trajectory not found for sequence {seq} in dataset {dataset_name}.")

            # 4.6 save sequence metrics to csv
            seq_metrics = {
                "dataset": dataset_name,
                "seq": seq,
                "ATE": ate,
                "RPE trans": rpe_trans,
                "RPE rot": rpe_rot,
            }
            write_csv(osp.join(output_root, "seq_metrics.csv"), seq_metrics)
            results.append((seq, ate, rpe_trans, rpe_rot))

            # 4.7. update metric for a sequence to tqdm bar
            tbar.set_postfix_str(f"Seq {seq} ATE: {ate:5.2f} | RPE-trans: {rpe_trans:5.2f} | RPE-rot: {rpe_rot:5.2f}")

        avg_ate, avg_rpe_trans, avg_rpe_rot = calculate_averages(results)

        dataset_metrics = {
            "ATE": avg_ate,
            "RPE trans": avg_rpe_trans,
            "RPE rot": avg_rpe_rot,
        }
        statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric")  # + ".csv"
        if getattr(hydra_cfg, "save_suffix", None) is not None:
            statistics_file += f"-{hydra_cfg.save_suffix}"
        statistics_file += ".csv"
        write_csv(statistics_file, dataset_metrics)
        logger.info(f"{dataset_name} - Average pose estimation metrics: {dataset_metrics}")
    
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    set_default_arg("evaluation", "relpose-distance")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    main()