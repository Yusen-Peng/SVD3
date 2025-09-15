# π³ Training Code

Welcome to the official training code for the π³ project.

This document provides a comprehensive guide to training the π³ model. We've strived to make this code robust and accurate, but issues may still exist. We highly appreciate any bug reports or suggestions for improvement. Please feel free to open an issue in this repository to share your feedback.


## 💾 Data Preparation

We provide three example datasets to get you started. Before you begin, **please ensure the `data_root` path in the corresponding dataloader script (located in the `datasets/` directory) is correctly set to your data's location.**

### Data Sampling Strategy

We provide three example data loaders, each designed with a specific sampling strategy to handle different datasets effectively.

The core philosophy is to create training batches with frames that are neither too dense nor too sparse. This is achieved by primarily sampling frames at intervals while occasionally including consecutive frames to ensure model robustness.

Here is a summary of the strategies:

  * **For small-scale indoor scenes**, frames are randomly sampled from the entire sequence.
  * **For large-scale indoor scenes or long sequences**, a dynamic sliding window approach is used. Frames are typically sampled randomly within this window. To ensure wider coverage, the window is occasionally divided into sub-intervals, with one frame sampled from each. There's also a small chance of sampling frames from the entire scene to capture global context.


## 🚀 Training Process

Our model is trained in three sequential stages. Please follow the steps in order.

> **Note:** The command below is an example using 8 GPUs. You should adjust the number of GPUs (`num_processes`) and training epochs based on your specific dataset and hardware.

### Stage 1: Low-Resolution Training

This initial stage trains the model on low-resolution data to establish a foundational checkpoint.

**Command:**

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
--num_processes 8 --num_machines 1 \
scripts/train_pi3.py train=train_pi3_lowres name=pi3_lowres
```

### Stage 2: High-Resolution Training

This stage fine-tunes the model on high-resolution data, loading the weights from the previous stage.

**Command:**

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
--num_processes 8 --num_machines 1 \
scripts/train_pi3.py train=train_pi3_highres name=pi3_highres \
model.ckpt=<path-to-lowres-checkpoint>
```

> **Note:** Please replace `<path-to-lowres-checkpoint>` with the actual path to the checkpoint file saved from Stage 1.

### Stage 3: Confidence Branch Training

This final stage trains the confidence prediction branch of the model.

**Prerequisite:**
Before running the command, you must download the Segformer pre-trained checkpoint.

  * **Download Link:** [segformer.b0.512x512.ade.160k.pth](https://github.com/NVlabs/SegFormer?tab=readme-ov-file) (from the OneDrive link for evaluation in README)
  * **Required Path:** Place the downloaded file at `ckpts/segformer.b0.512x512.ade.160k.pth`.

**Command:**

```bash
accelerate launch --config_file configs/accelerate/ddp.yaml \
--num_processes 8 --num_machines 1 \
scripts/train_pi3.py train=train_pi3_conf name=pi3_conf \
model.ckpt=<path-to-highres-checkpoint>
```

> **Note:** Please replace `<path-to-highres-checkpoint>` with the actual path to the checkpoint file saved from Stage 2.


## How to Reduce GPU Memory Usage
To reduce GPU memory consumption, adjust the following parameters in the recommended order of priority:

1. `train.max_img_per_gpu`: Decrease this value to reduce the number of images per batch on each GPU.

2. `train.pixel_count_range`: Lower the range to process images with a smaller pixel count.

3. `model.num_dec_blk_not_to_checkpoint`: Increase this value to apply activation checkpointing to more decoder blocks, which saves memory at the cost of some computational speed.