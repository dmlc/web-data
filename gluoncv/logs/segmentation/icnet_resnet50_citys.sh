#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./scripts/segmentation/train.py \
	--dataset citys \
	--model icnet \
	--backbone resnet50 \
	--syncbn \
	--ngpus 8 \
	--lr 0.01 \
	--epochs 240 \
	--base-size 2048 \
	--crop-size 768 \
	--workers 32 \
