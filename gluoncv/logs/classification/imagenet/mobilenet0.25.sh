python train_imagenet.py \
  --rec-train /media/ramdisk/rec/train.rec --rec-train-idx /media/ramdisk/rec/train.idx \
  --rec-val /media/ramdisk/rec/val.rec --rec-val-idx /media/ramdisk/rec/val.idx \
  --model mobilenet0.25 --mode hybrid \
  --lr 0.4 --lr-mode cosine --num-epochs 120 --batch-size 128 --num-gpus 8 -j 60 \
  --use-rec --dtype float16 --warmup-epochs 5 --no-wd --label-smoothing \
  --save-dir params_mobilenet0.25 \
  --logging-file mobilenet0.25.log
