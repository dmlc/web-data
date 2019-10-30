#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./scripts/action-recognition/train_recognizer.py \
	--dataset hmdb51 \
    --data-dir /home/ubuntu/.mxnet/datasets/hmdb51/rawframes \
    --train-list /home/ubuntu/.mxnet/datasets/hmdb51/testTrainMulti_7030_splits/hmdb51_train_split_1_rawframes.txt \
    --val-list /home/ubuntu/.mxnet/datasets/hmdb51/testTrainMulti_7030_splits/hmdb51_val_split_1_rawframes.txt \
	--model resnet50_v1b_hmdb51 \
	--mode hybrid \
	--dtype float32 \
    --prefetch-ratio 1.0 \
	--num-classes 51 \
	--batch-size 32 \
	--num-gpus 8 \
	--num-data-workers 32 \
	--new-height 256 \
	--new-width 340 \
	--new-length 1 \
	--new-step 1 \
	--input-size 224 \
	--lr-mode step \
	--lr 0.01 \
	--lr-decay 0.1 \
	--lr-decay-epoch 15,25,35 \
	--momentum 0.9 \
	--wd 0.0001 \
	--num-epochs 35 \
	--scale-ratios 1.0,0.8 \
	--save-frequency 5 \
	--clip-grad 40 \
	--log-interval 50 \
	--logging-file resnet50_v1b_hmdb51.log \
	--save-dir ./logs/ \


