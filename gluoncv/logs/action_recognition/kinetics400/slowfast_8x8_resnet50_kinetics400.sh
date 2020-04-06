#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./scripts/action-recognition/train_recognizer.py \
    --dataset kinetics400 \
    --data-dir /home/ubuntu/data/kinetics400/train_256 \
    --val-data-dir /home/ubuntu/data/kinetics400/val_256 \
    --train-list /home/ubuntu/data/kinetics400/k400_train_240618.txt \
    --val-list /home/ubuntu/data/kinetics400/k400_val_19761_cleanv3.txt \
    --dtype float32 \
    --mode hybrid \
    --prefetch-ratio 1.0 \
    --model slowfast_8x8_resnet50_kinetics400 \
    --slowfast \
    --slow-temporal-stride 8 \
    --fast-temporal-stride 2 \
    --video-loader \
    --use-decord \
    --num-classes 400 \
    --batch-size 8 \
    --num-gpus 8 \
    --num-data-workers 32 \
    --input-size 224 \
    --new-height 256 \
    --new-width 340 \
    --new-length 64 \
    --new-step 1 \
    --lr-mode cosine \
    --lr 0.11 \
    --momentum 0.9 \
    --wd 0.0001 \
    --num-epochs 196 \
    --warmup-epochs 34 \
    --warmup-lr 0.01 \
    --scale-ratios 1.0,0.8 \
    --save-frequency 20 \
    --log-interval 50 \
    --logging-file slowfast_8x8_resnet50_kinetics400.log
