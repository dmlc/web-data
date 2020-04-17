#!/usr/bin/env bash

python ./scripts/action-recognition/train_recognizer.py \
	--dataset ucf101 \
	--model inceptionv3_ucf101 \
	--num-classes 101 \
	--mode hybrid \
	--dtype float32 \
	--batch-size 25 \
	--num-gpus 8 \
	--num-data-workers 32 \
	--new-height 340 \
	--new-width 450 \
	--input-size 299 \
	--lr-mode step \
	--lr 0.001 \
	--momentum 0.9 \
	--wd 0.0005 \
	--lr-decay 0.1 \
	--lr-decay-epoch 30,60,80 \
	--num-epochs 80 \
	--clip-grad 40 \
	--partial-bn \
	--save-frequency 5 \
	--log-interval 10 \

