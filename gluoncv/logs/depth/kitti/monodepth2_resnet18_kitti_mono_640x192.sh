#!/usr/bin/env bash

python train.py --model_zoo monodepth2_resnet18_kitti_mono_640x192 --model_zoo_pose monodepth2_resnet18_posenet_kitti_mono_640x192 --pretrained_base --log_dir ./tmp/mono/ --png --gpu 0 --batch_size 10
