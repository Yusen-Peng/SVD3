import os
import json
import torch
import numpy as np
import open3d as o3d
import os.path as osp
import hydra
import torch.nn as nn
import logging
from safetensors.torch import load_file
from omegaconf import DictConfig, ListConfig

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from pi3.models.pi3 import Pi3
from vggt.models.vggt import VGGT
from utils.interfaces import infer_mv_pointclouds, adaptive_infer_mv_pointclouds, learn_entropy_cfg_from_calib, learn_drift_cfg_from_calib, drifting_adaptive_infer_mv_pointclouds, learn_augmented_entropy_cfg_from_calib, augmented_adaptive_infer_mv_pointclouds
from utils.interfaces import learn_entropy_cfg_continuous_from_calib, fine_grained_adaptive_infer_mv_pointclouds
from utils.interfaces import infer_mv_pointclouds_VGGT
from mv_recon.utils import umeyama, accuracy, completion
from utils.messages import set_default_arg, write_csv
from utils.vis_utils import save_image_grid_auto
from utils.interfaces import install_twofactor_modules_from_sd, strip_factor_keys, install_slicabletwofactor_modules_from_sd

from utils.interfaces import vggt_install_twofactor_modules_from_sd, vggt_install_slicabletwofactor_modules_from_sd
from utils.interfaces import adaptive_infer_mv_pointclouds_VGGT


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

        ADAPTIVE = 'BASE' in pretrained_model_name_or_path
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


    logger = logging.getLogger("mv_recon-eval")
    logger.info(f"Loaded Pi3 from {pretrained_model_name_or_path}")

    for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
        # 1.1 look up dataset config from configs/data, decide the dataset name, and load the dataset
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
        dataset_info = all_data_info[dataset_name]
        dataset = hydra.utils.instantiate(dataset_info.cfg)

        # 1.2 ready for output directory & metrics
        output_root = osp.join(hydra_cfg.output_dir, dataset_name)
        os.makedirs(output_root, exist_ok=True)
        all_data_dict = {
            "Acc-mean":  0.0,  "Acc-med":  0.0,
            "Comp-mean": 0.0,  "Comp-med": 0.0,
            "NC-mean":   0.0,  "NC-med":   0.0,
            "NC1-mean":  0.0,  "NC1-med":  0.0,
            "NC2-mean":  0.0,  "NC2-med":  0.0,
        }

        # 1.3 load pre-sampled seq-id-map
        logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Evaluating Multi-View Pointcloud Reconstruction of Pi3 on dataset {dataset_name}...")
        sample_config: DictConfig = dataset_info.sampling
        logger.info(f"Sampling strategy: {sample_config.strategy}")
        with open(dataset_info.seq_id_map, "r") as f:
            seq_id_map: dict = json.load(f)

        if osp.exists(osp.join(output_root, "_all_samples.csv")):
            os.remove(osp.join(output_root, "_all_samples.csv"))  # remove old csv file
        for seq_idx, (seq_name, ids) in enumerate(seq_id_map.items(), start=1):
            # 2. load data, choose specific ids of a sequence
            data = dataset.get_data(sequence_name=seq_name, ids=ids)
            filelist: list         = data['image_paths']  # [str] * N
            images: torch.Tensor   = data['images']       # (N, 3, H, W)
            gt_pts: np.ndarray     = data['pointclouds']  # (N, H, W, 3)
            valid_mask: np.ndarray = data['valid_mask']   # (N, H, W)

            # 3. real inference, predicted pointcloud aligned to ground truth (data_h, data_w)
            data_h, data_w         = images.shape[-2:]
            if COMPRESSED and ADAPTIVE and not USE_VGGT:
                if ADAPTIVE_MODE == 'input':
                    if not AUGMENTED:
                        if not FINE_GRAINED:
                            pred_pts: np.ndarray = adaptive_infer_mv_pointclouds(filelist, model, save_path, hydra_cfg, (data_h, data_w))  # (N, H, W, 3)
                        else:
                            pred_pts: np.ndarray = fine_grained_adaptive_infer_mv_pointclouds(filelist, model, hydra_cfg, (data_h, data_w))  # (N, H, W, 3)
                    else:
                        pred_pts: np.ndarray = augmented_adaptive_infer_mv_pointclouds(filelist, model, hydra_cfg, (data_h, data_w))  # (N, H, W, 3)
                elif ADAPTIVE_MODE == 'drift':
                    pred_pts: np.ndarray = drifting_adaptive_infer_mv_pointclouds(filelist, model, hydra_cfg, (data_h, data_w))  # (N, H, W, 3)
            
            else:
                if USE_VGGT:
                    if not ADAPTIVE:
                        pred_pts: np.ndarray = infer_mv_pointclouds_VGGT(filelist, model, hydra_cfg, (data_h, data_w))  # (N, H, W, 3)

                    else:
                        pred_pts: np.ndarray = adaptive_infer_mv_pointclouds_VGGT(filelist, model, save_path, hydra_cfg, (data_h, data_w))
                else:
                    pred_pts: np.ndarray = infer_mv_pointclouds(filelist, model, hydra_cfg, (data_h, data_w))  # (N, H, W, 3)
            assert pred_pts.shape == gt_pts.shape, f"Predicted points shape {pred_pts.shape} does not match ground truth shape {gt_pts.shape}."

            # 4. save input images
            seq_name = seq_name.replace("/", "-")
            save_image_grid_auto(images, osp.join(output_root, f"{seq_name}.png"))
            colors = images.permute(0, 2, 3, 1)[valid_mask].cpu().numpy().reshape(-1, 3)

            # 5. coarse align
            c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
            pred_pts = c * np.einsum('nhwj, ij -> nhwi', pred_pts, R) + t.T

            # 6. filter invalid points
            pred_pts = pred_pts[valid_mask].reshape(-1, 3)
            gt_pts = gt_pts[valid_mask].reshape(-1, 3)

            # 7. save predicted & ground truth point clouds
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pred_pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name}-pred.ply"), pcd)

            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
            pcd_gt.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(osp.join(output_root, f"{seq_name}-gt.ply"), pcd_gt)

            # 8. ICP align refinement
            if "DTU" in dataset_name:
                threshold = 100
            else:
                threshold = 0.1

            trans_init = np.eye(4)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd,
                pcd_gt,
                threshold,
                trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )

            transformation = reg_p2p.transformation
            pcd = pcd.transform(transformation)
            
            # 9. estimate normals
            pcd.estimate_normals()
            pcd_gt.estimate_normals()
            pred_normal = np.asarray(pcd.normals)
            gt_normal = np.asarray(pcd_gt.normals)

            # o3d.io.write_point_cloud(
            #     os.path.join(
            #         save_path, f"{seq.replace('/', '_')}-mask-icp.ply"
            #     ),
            #     pcd,
            # )

            # 10. compute metrics
            acc, acc_med, nc1, nc1_med = accuracy(
                pcd_gt.points, pcd.points, gt_normal, pred_normal
            )
            comp, comp_med, nc2, nc2_med = completion(
                pcd_gt.points, pcd.points, gt_normal, pred_normal
            )
            logger.info(
                f"[{dataset_name} {seq_idx}/{len(dataset.sequence_list)}] Seq: {seq_name}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
            )

            # 11. save metrics to csv
            write_csv(osp.join(output_root, f"_all_samples.csv"), {
                "seq":       seq_name,
                "Acc-mean":  acc,
                "Acc-med":   acc_med,
                "Comp-mean": comp,
                "Comp-med":  comp_med,
                "NC1-mean":  nc1,
                "NC1-med":   nc1_med,
                "NC2-mean":  nc2,
                "NC2-med":   nc2_med,
            })
            all_data_dict["Acc-mean"]  += acc
            all_data_dict["Acc-med"]   += acc_med
            all_data_dict["Comp-mean"] += comp
            all_data_dict["Comp-med"]  += comp_med
            all_data_dict["NC-mean"]   += (nc1 + nc2) / 2
            all_data_dict["NC-med"]    += (nc1_med + nc2_med) / 2
            all_data_dict["NC1-mean"]  += nc1
            all_data_dict["NC1-med"]   += nc1_med
            all_data_dict["NC2-mean"]  += nc2
            all_data_dict["NC2-med"]   += nc2_med

            # release cuda memory
            torch.cuda.empty_cache()

        num_samples = len(dataset)
        metric_dict = {
            metric: value / num_samples
            for metric, value in all_data_dict.items()
            if metric != "model"
        }

        statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric")  # + ".csv"
        if getattr(hydra_cfg, "save_suffix", None) is not None:
            statistics_file += f"-{hydra_cfg.save_suffix}"
        statistics_file += ".csv"
        write_csv(statistics_file, metric_dict)
    
    del model
    torch.cuda.empty_cache()
    logger.info(f"Finished evaluating Pi3 on all datasets.")


if __name__ == "__main__":
    set_default_arg("evaluation", "mv_recon")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()