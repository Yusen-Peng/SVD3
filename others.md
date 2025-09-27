# other stuff

## data example

Sample finetuning data:
```csharp
{'img': <PIL.Image.Image image mode=RGB size=224x224 at 0x7FD0EA5A85B0>, 'depthmap': array([[7.3085938, 0.       , 0.       , ..., 0.       , 0.       ,
        0.       ],
       [7.5234375, 0.       , 0.       , ..., 0.       , 0.       ,
        0.       ],
       [6.8984375, 0.       , 0.       , ..., 0.       , 0.       ,
        0.       ],
       ...,
       [9.3046875, 9.3046875, 9.3046875, ..., 9.21875  , 9.21875  ,
        9.2265625],
       [9.2890625, 9.2890625, 9.2890625, ..., 9.203125 , 9.203125 ,
        9.2109375],
       [9.265625 , 9.28125  , 9.28125  , ..., 9.1875   , 9.1875   ,
        9.1875   ]], dtype=float32), 'camera_pose': array([[-1., -0., -0., -0.],
       [-0., -1., -0., -0.],
       [ 0.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  1.]], dtype=float32), 'camera_intrinsics': array([[126.97792,   0.     , 111.68296],
       [  0.     , 126.97792, 111.68297],
       [  0.     ,   0.     ,   1.     ]], dtype=float32), 'dataset': 'COD3DV2', 'label': 'toytrain-240_25394_51994', 'instance': 'frame000008.jpg'}
```



## SVD-LLM preliminaries

### Truncation-Aware Data Whitening

using the calibration dataset for data whitening:

```bash
CUDA_VISIBLE_DEVICES=<whichever_is_free> python SVDLLM.py --model jeffwan/llama-7b-hf --step 1 --ratio 0.2 --whitening_nsamples 256 --dataset wikitext2 --seed 3 --model_seq_len 2048 --save_path . --run_low_resource
```

perplexity evaluation:

```bash
CUDA_VISIBLE_DEVICES=<whichever_is_free> taskset -c 30-40 python SVDLLM.py --step 4 --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt
```

```java
PPL after pruning: {'wikitext2': 7.886700954800093}
Weight Memory: 22004.896484375 MiB
```

efficiency evaluation:

```bash
CUDA_VISIBLE_DEVICES=<whichever_is_free> taskset -c 30-40 python SVDLLM.py --step 5 --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt
```

```java
Total Memory: 28.538090705871582 GB
Weight Memory: 20.503570556640625 GB
Activation Memory: 8.026554107666016 GB
Throughput: 69.48256829185354 tokens/sec
```

### Finetuning with LoRA

update W'u (~35 hours):

```bash
CUDA_VISIBLE_DEVICES=<whichever_is_free> nohup taskset -c 30-40 python utils/LoRA.py --prune_model jeffwan_llama_7b_hf_whitening_only_0.8.pt --data_path yahma/alpaca-cleaned --output_dir ./first_half --lora_target_modules q_u_proj,k_u_proj,v_u_proj,o_u_proj,gate_u_proj,down_u_proj,up_u_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 4 --micro_batch_size 1 --cutoff_len 1024 --group_by_length &
```

```java
{'train_runtime': 128586.2054, 'train_samples_per_second': 1.161, 'train_steps_per_second': 0.29, 'train_loss': 1.0868874290876194, 'epoch': 3.0}
```

Immediate evaluation:

```bash
CUDA_VISIBLE_DEVICES=<whichever_is_free> taskset -c 30-40 python SVDLLM.py --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt --lora ./first_half --step 4
```

```java
PPL after pruning: {'wikitext2': 7.23282080013519}
Weight Memory: 22004.896484375 MiB
```

Update W'v:

```bash
CUDA_VISIBLE_DEVICES=<whichever_is_free> nohup taskset -c 30-40 python utils/LoRA.py --prune_model ./first_half/merge.pt --data_path yahma/alpaca-cleaned --output_dir ./second_half --lora_target_modules q_v_proj,k_v_proj,v_v_proj,o_v_proj,gate_v_proj,down_v_proj,up_v_proj --lora_r 8 --num_epochs 3 --learning_rate 1e-4 --batch_size 4 --micro_batch_size 1 --cutoff_len 1024 --group_by_length &
```

Immediate evaluation:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 taskset -c 30-40 python SVDLLM.py --model_path jeffwan_llama_7b_hf_whitening_only_0.8.pt --lora ./first_half /first_half --step 4
```

```java
coming soon!
```

Final evaluation:

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 taskset -c 30-40 python SVDLLM.py --model_path ./first_half/merge.pt --lora ./second_half --step 4
```

## Efficiency Measurement

HuggingFace repo for π3 model: [yyfz233/Pi3](https://huggingface.co/yyfz233/Pi3)

get forward pass (ms) and throughput (frames/sec):

```bash
CUDA_VISIBLE_DEVICES=0 python example.py --ckpt ./pi3_model.safetensors --efficiency_measure simple
```

| model | forward pass (ms) | throughput (frames/sec) |
| ----- | ----------------- | ----------------------- |
| original π3 | 1530.90 | 7.19 |
| SVD-π3 (coming soon!) | ? | ? |

detailed profiling:

```bash
CUDA_VISIBLE_DEVICES=0 python example.py --ckpt ./pi3_model.safetensors --efficiency_measure profiler
```

<img src="Pi3_main/topk_cuda_ops.png" width=500 height=400></img>

## Evaluation

### Monocular Depth Estimation

dataset collection:

- [x] Sintel
- [ ] Bonn
- [ ] KITTI
- [ ] NYU-v2

Sintel dataset:
| model | Abs Rel | Sq Rel | RMSE | Log RMSE |
| ----- | ------ | ---- | ----- | ------ |
| original π3 | 0.2796 | 1.2767 | 3.7178 | 0.5286 | 
| SVD-π3 (coming soon!) | ? | ? | ? | ? |

### Video Depth Estimation

dataset collection:

- [x] Sintel
- [ ] Bonn
- [ ] KITTI
- [ ] NYU-v2

Sintel dataset:
| model | Abs Rel | Sq Rel | RMSE | Log RMSE |
| ----- | ------ | ---- | ----- | ------ |
| original π3 | 0.2106 | 1.2873 | 4.0003 | 0.4840 | 
| SVD-π3 (coming soon!) | ? | ? | ? | ? |


### Relative Camera Pose Estimation

dataset collection:

- [ ] RealEstate10K (too big for now; TB level)
- [x] Sintel
- [ ] TUM-dynamics
- [ ] ScanNetv2

Sintel dataset:
| model | ATE | RPE trans | RPE rot |
| ----- | ------ | ---- | ----- |
| original π3 | 0.0732 | 0.0390 | 0.2766 |
| SVD-π3 (coming soon!) | ? | ? | ? |


### Point Map Estimation

dataset collection:

- [x] 7-Scenes
  - [x] preprocessing
- [ ] Neural-NRGBD
- [ ] DTU

7-scenes-dense:

| model | Acc-mean | Acc-med | Comp-mean | Comp-med | NC-mean | NC-med | NC1-mean | NC1-med | NC2-mean | NC2-med |
| ----- | -------- | ------- | --------- | -------- | ------- | ------ | -------- | ------- | -------- | ------- | 
| original π3 | 0.0157 | 0.0066 | 0.0220 | 0.0105 | 0.6886 | 0.7920 | 0.6913 | 0.7975 | 0.6859 | 0.7865 |
| SVD-π3 (coming soon!) | ? | ? | ? | ?  | ? | ? |  ? |  ? |  ? | ? |  

7-scenes-sparse:

| model | Acc-mean | Acc-med | Comp-mean | Comp-med | NC-mean | NC-med | NC1-mean | NC1-med | NC2-mean | NC2-med |
| ----- | -------- | ------- | --------- | -------- | ------- | ------ | -------- | ------- | -------- | ------- | 
| original π3 | 0.0469 | 0.0284 | 0.0736 | 0.0484 | 0.7413 | 0.8402 | 0.7446 | 0.8427 | 0.7379 | 0.8378 |
| SVD-π3 (coming soon!) | ? | ? | ? | ?  | ? | ? |  ? |  ? |  ? | ? |  
