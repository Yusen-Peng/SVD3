#!/bin/bash

## Monocular Depth Estimation
PYTHONNOUSERSITE=1 python Pi3_evaluation/monodepth/infer.py
PYTHONNOUSERSITE=1 python Pi3_evaluation/monodepth/eval.py

## Video Depth Estimation
PYTHONNOUSERSITE=1 python Pi3_evaluation/videodepth/infer.py
PYTHONNOUSERSITE=1 python Pi3_evaluation/videodepth/eval.py

## camera-distance
PYTHONNOUSERSITE=1 python Pi3_evaluation/relpose/eval_dist.py

# point cloud
PYTHONNOUSERSITE=1 python Pi3_evaluation/mv_recon/eval.py
