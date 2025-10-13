## 🔥 SVD-π3 Implementation Roadmap

- [x] Truncation-Aware Data Whitening
  - [x] collect calibration data
    - [x] sintel_training_ALLMODS_512_224_8_10_3.pt (64 batches, 512 images)
  - [x] derive the whitening matrix via profiling 
    - [x] **fallback to EVD when Cholesky fails ("✅95/144 succeeded with Cholesky, 49/144 used EVD fallback")**
  - [x] apply whitening
    - [x] SVD_Pi3Attention
    - [x] SVD_Pi3MLP
    - [x] Pi3_whitening
      - [x] **hierarchical attempts on SVD (float32 GPU -> float64 GPU -> float64 CPU)**
- [x] evaluation
  - [x] performance/accuracy evaluation (for now, focus on depth estimation)
    - [x] in order to load checkpoints, implement a CompressedPi3 that inherits Pi3 
    - [x] load the (whitened + compressed) checkpoints
    - [x] run evaluation
  - [x] efficiency evaluation
    - [x] throughput (img/sec)
- [x] LoRA finetuning
  - [x] implement Pi3TrainerLoRA from Pi3Trainer [Pi3_main/trainers/pi3_trainer.py](Pi3_main/trainers/pi3_trainer.py)
      - [x] wrap CompressedPi3 with LoRA
  - [x] configure finetuning hyperparameters in [Pi3_main/Pi3_LoRA_helper.py](Pi3_main/Pi3_LoRA_helper.py)
  - [x] dataset preparation
    - [x] download CO3Dv2 dataset (single-sequence version: 88 sequences in total)
    - [x] train/test split (manual implementation)
    - [x] sometimes depth map is all zeros (I make constant/dummy depth map accordingly)
  - [x] LoRA Training 
    - [x] lora on W'u
    - [x] lora on W'v

## Truncation-aware data whitening

```bash
# stay in 'SVD-pi3' (root directory)
CUDA_VISIBILE_DEVICES=0 python Pi3_main/SVDPi3.py --step 1 --ckpt Pi3_main/pi3_model.safetensors --save_path /data/wanghaoxuan/SVD_Pi3_cache
```

## LoRA finetuning

```bash
# stay in 'SVD-pi3' (root directory)
accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 1 --num_machines 1 Pi3_main/Pi3_LoRA.py --prune_model /data/wanghaoxuan/SVD_Pi3_cache/Pi3_whitening_only_0.8.safetensors --num_epochs 3 --batch_size 4 --micro_batch_size 1 --learning_rate 1e-4 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05
```

## Evaluation

- [x] Monocular Depth Estimation (Sintel dataset for now)

```bash
# stay in 'SVD-pi3' (root directory)
python Pi3_evaluation/monodepth/infer.py
python Pi3_evaluation/monodepth/eval.py
```

original π3:

```python
{'Abs Rel': 0.2796176944859495, 'Sq Rel': 1.276705312700384, 'RMSE': 3.7178159584027632, 'Log RMSE': 0.5286988564434099, 'δ < 1.': 0.0, 'δ < 1.25': 0.616369625758487, 'δ < 1.25^2': 0.78750964094564, 'δ < 1.25^3': 0.8520245890568844}
```

```log
[2025-09-15 19:56:42,655][monodepth-infer][INFO] - Overall throughput: 6.66 images/second
```

compressed π3 (no LoRA finetuned):

```python
{'Abs Rel': 0.7001590852341619, 'Sq Rel': 5.015428265738036, 'RMSE': 7.187596822293912, 'Log RMSE': 0.6972085129552443, 'δ < 1.': 0.0, 'δ < 1.25': 0.3193632831349897, 'δ < 1.25^2': 0.5601810225270805, 'δ < 1.25^3': 0.7066331125178978}
```

```log
[2025-09-15 19:52:35,451][monodepth-infer][INFO] - Overall throughput: 7.75 images/second
```

compressed π3 (+ LoRA finetuned on both W'u and W'v with 3 epochs):

```python
{'Abs Rel': 0.7253016288184739, 'Sq Rel': 5.290863304825592, 'RMSE': 7.349976061546238, 'Log RMSE': 0.7360445591543378, 'δ < 1.': 0.0, 'δ < 1.25': 0.28735444342269517, 'δ < 1.25^2': 0.522024572542984, 'δ < 1.25^3': 0.6837374942715188}
```

```log
[2025-09-27 08:00:46,789][monodepth-infer][INFO] - Overall throughput: 10.23 images/second
```

- [ ] Video Depth Estimation

- [ ] Relative Camera Pose Estimation

- [ ] Point Map Estimation


## SVD-π3 Pipeline