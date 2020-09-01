#!/usr/bin/env bash

python train.py --model_zoo monodepth2_resnet18_kitti_stereo_640x192 --pretrained_base --split eigen_full --frame_ids 0 --use_stereo --log_dir ./tmp/stereo/ --png --gpu 0 --batch_size 12
