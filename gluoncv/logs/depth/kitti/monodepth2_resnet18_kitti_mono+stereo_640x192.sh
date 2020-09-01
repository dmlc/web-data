#!/usr/bin/env bash

python train.py --model_zoo monodepth2_resnet18_kitti_mono+stereo_640x192 --model_zoo_pose monodepth2_resnet18_posenet_kitti_mono+stereo_640x192 --pretrained_base --frame_ids 0 -1 1 --use_stereo --log_dir ./tmp/mono_stereo/ --png --gpu 0 —batch_size 8
