#!/usr/bin/env bash

# cmd for training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./scripts/segmentation/train.py \
    --dataset citys \
    --model danet \
    --backbone resnet101 \
    --syncbn \
    --ngpus 8 \
    --lr 0.01 \
    --epochs 240 \
    --base-size 2048 \
    --crop-size 768 \
    --workers 32 \

# cmd for evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./scripts/segmentation/test.py \
    --model-zoo danet_resnet101_citys \
    --dataset citys \
    --batch-size 8 \
    --ngpus 8 \
    --eval \
    --pretrained \
    --base-size 2048 \
    --crop-size 768
