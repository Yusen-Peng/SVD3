# SVD-π3: Efficient Visual Geometry Learning via Truncation-aware Singular Value Decomposition

![alt text](docs/SVD_Pi3.png)

## Plain SVD baseline

```bash
# stay in 'SVD-pi3' (root directory)
CUDA_VISIBLE_DEVICES=0 python Pi3_main/SVDPi3.py --ckpt /data/wanghaoxuan/SVD_Pi3_cache/model.safetensors --save_path /data/wanghaoxuan/SVD_Pi3_cache --ratio 0.2 --baseline
```

## Truncation-aware data whitening

```bash
# stay in 'SVD-pi3' (root directory)
CUDA_VISIBLE_DEVICES=0 python Pi3_main/SVDPi3.py --ckpt /data/wanghaoxuan/SVD_Pi3_cache/model.safetensors --save_path /data/wanghaoxuan/SVD_Pi3_cache --ratio 0.2 --calibration_dataset_path /data/wanghaoxuan/sintel --whitening_nsamples 256
# or use scannetv2 as calibration dataset
CUDA_VISIBLE_DEVICES=0 python Pi3_main/SVDPi3.py --ckpt /data/wanghaoxuan/SVD_Pi3_cache/model.safetensors --save_path /data/wanghaoxuan/SVD_Pi3_cache --ratio 0.2 --calibration_dataset_path /data/wanghaoxuan/scannetv2 --whitening_nsamples 256
```

### Detailed illustration

![alt text](/plain_whitening.png)

### Cholesky Decomposition does not always succeed

```bash
# collect the whitening matrix
✅56/144 succeeded with Cholesky, 88/144 used EVD fallback
```

### Whitening is always applied successfully

```bash
# apply whitening
✅ 1 out of 144 layers fell back to plain SVD without whitening.✅
```

### Calibration dataset

from SVD-LLM paper: *"the changes of the three key characteristics on calibration data incur no more than 3% to the final performance, indicating that the sensitivity of SVD-LLM on calibration data is limited."*

from Pi3 paper: *"we train the model on a large-scale aggregation of 15 diverse datasets... include GTA-SfM, CO3D, WildRGB-D, Habita, ARK-itScenes, TartanAir, ScanNet, ScanNet++, BlendedMVG, MatrixCity, MegaDepth, Hypersim, Taskonomy, Mid-Air, and an internal dynamic scene dataset..."* and **we already have ScanNet-v2 and CO3D-v2 (single-seq) on the server**!

## Evaluation

- [x] Monocular Depth Estimation

```bash
# stay in 'SVD-pi3' (root directory)
python Pi3_evaluation/monodepth/infer.py
python Pi3_evaluation/monodepth/eval.py
```

- [x] Video Depth Estimation

```bash
# stay in 'SVD-pi3' (root directory)
python Pi3_evaluation/videodepth/infer.py
python Pi3_evaluation/videodepth/eval.py
```

- [ ] camera-angular (TBD, acquiring datasets is tricky...)

- [x] camera-distance

```bash
# stay in 'SVD-pi3' (root directory)
python Pi3_evaluation/relpose/eval_dist.py
```

- [x] point-map

```bash
# stay in 'SVD-pi3' (root directory)
python Pi3_evaluation/mv_recon/eval.py
# optional visualization
python point_cloud_visualization_7scenes.py # for 7scenes
python point_cloud_visualization_nrgbd.py # for NRGBD
```

## LoRA finetuning (WIP)

```bash
# a simple quick run
accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 1 --num_machines 1 Pi3_main/Pi3_LoRA.py --prune_model /data/wanghaoxuan/SVD_Pi3_cache/Pi3_whitening_only_scannet_0.2.safetensors --num_epochs 3 --batch_size 4 --micro_batch_size 1 --learning_rate 1e-4 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05
# run a long job offline
nohup accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 1 --num_machines 1 Pi3_main/Pi3_LoRA.py --prune_model /data/wanghaoxuan/SVD_Pi3_cache/Pi3_whitening_only_scannet_0.2.safetensors --num_epochs 3 --batch_size 4 --micro_batch_size 1 --learning_rate 1e-4 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 > lora_train_3epochs.log 2>&1 &
```
