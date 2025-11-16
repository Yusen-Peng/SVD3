# SVD-π3: Efficient Visual Geometry Learning via Singular Value Decomposition

![alt text](docs/SVD_Pi3.png)

## Truncation-aware data whitening

```bash
# stay in 'SVD-pi3' (root directory)
CUDA_VISIBLE_DEVICES=0 python Pi3_main/SVDPi3.py --ckpt /data/wanghaoxuan/SVD_Pi3_cache/model.safetensors --save_path /data/wanghaoxuan/SVD_Pi3_cache --ratio 0.3
```

## LoRA finetuning

```bash
# stay in 'SVD-pi3' (root directory)
accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 1 --num_machines 1 Pi3_main/Pi3_LoRA.py --prune_model /data/wanghaoxuan/SVD_Pi3_cache/Pi3_svd_baseline_0.6.safetensors --num_epochs 3 --batch_size 4 --micro_batch_size 1 --learning_rate 1e-4 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05
```

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

- [ ] camera-angular

- [x] camera-distance

```bash
# stay in 'SVD-pi3' (root directory)
python Pi3_evaluation/relpose/eval_dist.py
```

- [x] point-map

```bash
# stay in 'SVD-pi3' (root directory)
python Pi3_evaluation/mv_recon/eval.py
```
