python train_imagenet.py \
  --rec-train /media/ramdisk/rec/train.rec --rec-train-idx /media/ramdisk/rec/train.idx \
  --rec-val /media/ramdisk/rec/val.rec --rec-val-idx /media/ramdisk/rec/val.idx \
  --model resnet152_v1d --mode hybrid \
  --lr 0.1 --lr-mode cosine --num-epochs 200 --batch-size 32 --num-gpus 8 -j 60 \
  --use-rec --warmup-epochs 5 --last-gamma --no-wd --label-smoothing --mixup \
  --save-dir params_resnet152_v1d_mixup \
  --logging-file resnet152_v1d_mixup.log
