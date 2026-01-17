<p align="center">
<img src="docs/SVD_Pi3.png" width="850"/>
</p>

<h1 align="center">SVD-π3</h1>
<h2 align="center">Data-adaptive SVD for Efficient Visual Geometry Learning</h2>

[Full Results on Overleaf](https://www.overleaf.com/project/68d89c98e6991a1fc59ea65e)

## Data Adaptive SVD (entropy-guided; works fine 👍)

- [x] start with a *base* whitened + SVD-ed model with **40%** retention ratio
- [x] during inference, **slice** to 30%, 20%, 10% retention ratio based on input's entropy in the test set
  - slice: 
    - z = x @ V[:r].T
    - y = z @ U[:, :r].T + b
- [x] compute shannon entropy score **s(x)** and slice the rank (according to learned entropy thresholds)
- [x] using the calibration dataset, learn a mapping between entropy score s(x) and retention ratio (30% or 20% or 10%) while maintain **20%** retention ratio on average


learned ``adaptive_cfg.json``:

```JSON
{
  "entropy_p5": 0.6597831845283508,
  "entropy_p95": 5.533994674682617,
  "rr_values": [
    0.1,
    0.2,
    0.3
  ],
  "tail_frac": 0.25,
  "rr_thresholds": [
    0.27931690216064453,
    0.732358992099762
  ]
}
```

## Data Adaptive SVD (encoder-embedding; bad)

idea: instead of checking the shannon entropy of input images, do it on the embedding after encoder layers
- [x] apply k-means on embeddings (K=256) + compute entropy to allocate compression ratio (30%, 20%, 10%)  
- [x] integrate its pipeline into evaluation

results are bad though: 

on Kitti 0.2711 Abs Rel versus 0.0984 (input-entropy-guided)

potential explanation from ChatGPT: 
- *"Pixel entropy survives domain shift better because it’s low-level; embedding entropy is very domain-sensitive."*


## Data Adaptive SVD (early-layer-cos-sim-drift)

idea: compute cosine similarity drift in early (4) layers to allocate compression ratio (30%, 20%, 10%) 
- [x] cosine similarity drift scorer
- [x] hidden state extraction from early decoder layer
- [x] integrate it into the evaluation pipeline


learned ``adaptive_cfg_drifting.json``:

```JSON
{
  "score_p5": 0.06869048625230789,
  "score_p95": 0.07596379518508911,
  "rr_values": [
    0.1,
    0.2,
    0.3
  ],
  "tail_frac": 0.25,
  "drift_probe": {
    "kind": "cosine_drift_early_decoder",
    "probe_layers": 4,
    "ignore_special_tokens": true
  },
  "rr_thresholds": [
    0.37957727909088135,
    0.6646890640258789
  ]
}
```


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

<!-- ### Detailed illustration

![alt text](/plain_whitening.png) -->

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
