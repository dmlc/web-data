#!/usr/bin/env bash

python ./scripts/action-recognition/train_recognizer.py \
	--data-dir /home/ubuntu/yizhu/data/UCF101/rawframes \
	--train-list /home/ubuntu/yizhu/data/UCF101/ucfTrainTestlist/ucf101_train_split_1_rawframes.txt \
	--val-list /home/ubuntu/yizhu/data/UCF101/ucfTrainTestlist/ucf101_val_split_1_rawframes.txt \
	--model inceptionv3_ucf101 \
	--num-classes 101 \
	--mode hybrid \
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
	--logging-file 2d_rgb_inceptionv3_f1s1_b25_g8.txt \
	--save-dir /home/ubuntu/yizhu/logs/mxnet/pullrequest/2d_rgb_inceptionv3_f1s1_b25_g8 \

