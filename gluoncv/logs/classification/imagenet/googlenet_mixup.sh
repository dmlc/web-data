#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./scripts/classification/imagenet/train_imagenet.py \
  --rec-train /media/ramdisk/rec/train.rec --rec-train-idx /media/ramdisk/rec/train.idx \
  --rec-val /media/ramdisk/rec/val.rec --rec-val-idx /media/ramdisk/rec/val.idx \
  --model googlenet --mode hybrid --input-size 224 \
  --lr 0.4 --lr-mode cosine --num-epochs 200 --batch-size 128 --num-gpus 8 -j 60 \
  --use-rec --dtype float16 --warmup-epochs 5 --no-wd --label-smoothing --mixup \
  --save-dir ./params_googlenet_mixup \
  --logging-file googlenet_mixup.log

