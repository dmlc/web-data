#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./scripts/segmentation/train.py \
	--dataset mhpv1 \
	--model icnet \
	--backbone resnet50 \
	--syncbn \
	--ngpus 8 \
	--optimizer adam \
	--lr 0.00001 \
	--epochs 120 \
	--base-size 768 \
	--crop-size 768 \
	--workers 32 \
	--batch-size 16 \
	--log-interval 1
