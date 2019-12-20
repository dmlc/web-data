#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./scripts/segmentation/train.py \
    --dataset citys \
    --model psp \
    --aux \
    --backbone resnet101 \
    --syncbn \
    --ngpus 8 \
    --checkname psp_resnet101_citys \
    --lr 0.01 \
    --epochs 240 \
    --base-size 2048 \
    --crop-size 768 \
    --workers 48
