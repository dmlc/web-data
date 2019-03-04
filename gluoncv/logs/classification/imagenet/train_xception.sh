python train_imagenet.py \
  --rec-train /media/ramdisk/rec/train.rec --rec-train-idx /media/ramdisk/rec/train.idx \
  --rec-val /media/ramdisk/rec/val.rec --rec-val-idx /media/ramdisk/rec/val.idx \
  --model xception --mode hybrid --input-size 299 \
  --lr 0.1 --lr-mode cosine --num-epochs 200 --batch-size 32 --num-gpus 8 -j 60 \
  --use-rec --dtype float32 --warmup-epochs 5 --no-wd --label-smoothing \
  --save-dir params_xception \
  --logging-file xception.log
