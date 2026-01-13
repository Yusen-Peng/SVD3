# Data-adaptive SVD for Efficient Visual Geometry Learning

![alt text](docs/SVD_Pi3.png)

## Data Adaptive part (brainstorming)

- [x] start with a *base* whitened + SVD-ed model with **40%** retention ratio
- [ ] during inference, **slice** to 30%, 20%, 10% retention ratio based on input's entropy in the test set
  - slice: 
    - z = x @ V[:r].T
    - y = z @ U[:, :r].T + b
- [ ] compute entropy score s(x) for all samples from the calibration dataset to learn entropy thresholds
  - learn a mapping between entropy score s(x) and retention ratio (30%? 20%? 10%?)
- [ ] ideal outcomes: on average, we still maintain **20%** retention ratio in order to compare with baseline methods (i.e., direct SVD and homogeneous whitening) on all benchmarks


## Plain SVD baseline

```bash
# stay in 'SVD-pi3' (root directory)
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python Pi3_main/SVDPi3.py --ckpt /data/wanghaoxuan/SVD_Pi3_cache/model.safetensors --save_path /data/wanghaoxuan/SVD_Pi3_cache --ratio 0.2 --baseline
```

## Truncation-aware data whitening

```bash
# stay in 'SVD-pi3' (root directory)
CUDA_VISIBLE_DEVICES=0 pPYTHONNOUSERSITE=1 ython Pi3_main/SVDPi3.py --ckpt /data/wanghaoxuan/SVD_Pi3_cache/model.safetensors --save_path /data/wanghaoxuan/SVD_Pi3_cache --ratio 0.2 --calibration_dataset_path /data/wanghaoxuan/sintel --whitening_nsamples 256
# or use scannetv2 as calibration dataset (RECOMMENDED)
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python Pi3_main/SVDPi3.py --ckpt /data/wanghaoxuan/SVD_Pi3_cache/model.safetensors --save_path /data/wanghaoxuan/SVD_Pi3_cache --ratio 0.2 --calibration_dataset_path /data/wanghaoxuan/scannetv2 --whitening_nsamples 256
```

### Detailed illustration

![alt text](/plain_whitening.png)

### Cholesky Decomposition does not always succeed

```bash
# collect the whitening matrix
✅56/144 succeeded with Cholesky, 88/144 used EVD fallback
```

### Whitening is not always applied successfully

```bash
# apply whitening
✅ 1 out of 144 layers fell back to plain SVD without whitening.✅
```



<!-- ## LoRA finetuning (WIP)

```bash
# a simple quick run
accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 1 --num_machines 1 Pi3_main/Pi3_LoRA.py --prune_model /data/wanghaoxuan/SVD_Pi3_cache/Pi3_whitening_only_scannet_0.2.safetensors --num_epochs 3 --batch_size 4 --micro_batch_size 1 --learning_rate 1e-4 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05
# run a long job offline
nohup accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 1 --num_machines 1 Pi3_main/Pi3_LoRA.py --prune_model /data/wanghaoxuan/SVD_Pi3_cache/Pi3_whitening_only_scannet_0.2.safetensors --num_epochs 3 --batch_size 4 --micro_batch_size 1 --learning_rate 1e-4 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 > lora_train_3epochs.log 2>&1 &
``` -->