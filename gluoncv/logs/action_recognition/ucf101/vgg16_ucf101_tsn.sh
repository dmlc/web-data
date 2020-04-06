#!/usr/bin/env bash

python ./scripts/action-recognition/train_recognizer.py \
	--dataset ucf101 \
	--model vgg16_ucf101 \
	--num-classes 101 \
	--mode hybrid \
	--dtype float32 \
	--batch-size 25 \
	--num-segments 3 \
	--use-tsn \
	--num-gpus 8 \
	--num-data-workers 32 \
	--new-height 256 \
	--new-width 340 \
	--input-size 224 \
	--lr-mode step \
	--lr 0.001 \
	--momentum 0.9 \
	--wd 0.0005 \
	--lr-decay 0.1 \
	--lr-decay-epoch 30,60,80 \
	--num-epochs 80 \
	--save-frequency 5 \
	--log-interval 10 \
