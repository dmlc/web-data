#!/usr/bin/env bash

# cmd for training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./scripts/segmentation/train.py \
	--dataset citys \
	--model fastscnn \
	--aux \
	--ngpus 8 \
	--lr 0.045 \
	--epochs 1000 \
	--base-size 2048 \
	--crop-size 1024 \
	--workers 32 \
	--batch-size 32

# cmd for evaluation
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./scripts/segmentation/test.py \
	--model fastscnn \
	--dataset citys \
	--batch-size 8 \
	--ngpus 8 \
	--eval \
	--pretrained \
	--aux \
	--base-size 2048 \
	--height 1024 \
	--width 2048
