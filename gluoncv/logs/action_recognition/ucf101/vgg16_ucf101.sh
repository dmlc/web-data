#!/usr/bin/env bash

python ./scripts/action-recognition/train_recognizer.py \
	--data-dir /home/ubuntu/yizhu/data/UCF101/rawframes \
	--train-list /home/ubuntu/yizhu/data/UCF101/ucfTrainTestlist/ucf101_train_split_1_rawframes.txt \
	--val-list /home/ubuntu/yizhu/data/UCF101/ucfTrainTestlist/ucf101_val_split_1_rawframes.txt \
	--model vgg16_ucf101 \
	--num-classes 101 \
	--mode hybrid \
	--batch-size 25 \
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
	--logging-file 2d_rgb_vgg16_f1s1_b25_g8.txt \
	--save-dir /home/ubuntu/yizhu/logs/mxnet/2d_rgb_vgg16_f1s1_b25_g8 \
