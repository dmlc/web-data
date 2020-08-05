#!/usr/bin/env bash

python train.py --model_zoo monodepth2_resnet18_kitti_stereo_640x192 --pretrained_base --frame_ids 0 --use_stereo --split eigen_full --log_dir ./tmp/stereo_hybridize/ --png --gpu 0
