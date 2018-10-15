python train_imagenet.py \
  --rec-train /media/ramdisk/rec/train.rec --rec-train-idx /media/ramdisk/rec/train.idx \
  --rec-val /media/ramdisk/rec/val.rec --rec-val-idx /media/ramdisk/rec/val.idx \
  --model resnet152_v1c --mode hybrid \
  --lr 0.1 --lr-mode cosine --num-epochs 120 --batch-size 32 --num-gpus 8 -j 60 \
  --warmup-epochs 5 \
  --use-rec --last-gamma --no-wd --label-smoothing \
  --save-dir params_resnet152_v1c_best \
  --logging-file resnet152_v1c_best.log
