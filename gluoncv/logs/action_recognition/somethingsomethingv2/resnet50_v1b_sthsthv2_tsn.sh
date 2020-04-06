#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./scripts/action-recognition/train_recognizer.py \
	--dataset somethingsomethingv2 \
    --data-dir /home/ubuntu/third_disk/data/somethingsomethingv2/20bn-something-something-v2-frames \
    --train-list /home/ubuntu/third_disk/data/somethingsomethingv2/settings/train_videofolder.txt \
    --val-list /home/ubuntu/third_disk/data/somethingsomethingv2/settings/val_videofolder.txt \
	--model resnet50_v1b_sthsthv2 \
	--mode hybrid \
	--dtype float32 \
    --prefetch-ratio 1.0 \
	--num-classes 174 \
	--batch-size 16 \
	--num-segments 8 \
	--use-tsn \
	--num-gpus 8 \
	--num-data-workers 32 \
	--new-height 256 \
	--new-width 340 \
	--new-length 1 \
	--new-step 1 \
	--input-size 224 \
	--lr-mode step \
	--lr 0.01 \
	--momentum 0.9 \
	--wd 0.0001 \
	--lr-decay 0.1 \
	--lr-decay-epoch 10,20,30 \
	--num-epochs 30 \
	--scale-ratios 1.0,0.8 \
	--save-frequency 5 \
	--clip-grad 40 \
	--log-interval 50 \
	--logging-file resnet50_v1b_sthsthv2_tsn.log \
	--save-dir ./logs/ \

