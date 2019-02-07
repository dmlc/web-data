set -e
python3 ~/gluon-cv/scripts/classification/imagenet/train_imagenet.py \
  --rec-train /media/ramdisk/rec/train.rec --rec-train-idx /media/ramdisk/rec/train.idx \
  --rec-val /media/ramdisk/rec/val.rec --rec-val-idx /media/ramdisk/rec/val.idx \
  --model se_resnext50_32x4d --mode hybrid \
  --lr 0.4 --lr-mode cosine --num-epochs 200 --batch-size 128 --num-gpus 8 -j 90 \
  --use-rec --dtype float32 --warmup-epochs 5 --last-gamma --no-wd --label-smoothing --mixup \
  --save-dir params_se_resnext50_32x4d_best \
  --logging-file se_resnext50_32x4d_best.log
