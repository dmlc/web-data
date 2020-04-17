python train_imagenet.py \
  --rec-train /media/ramdisk/rec/train.rec --rec-train-idx /media/ramdisk/rec/train.idx \
  --rec-val /media/ramdisk/rec/val.rec --rec-val-idx /media/ramdisk/rec/val.idx \
  --model mobilenetv3_large --mode hybrid --wd 0.00003 \
  --lr 2.6 --lr-mode cosine --num-epochs 360 --batch-size 256 --num-gpus 8 -j 96 \
  --warmup-epochs 5 --dtype float16 --no-wd --last-gamma \
  --use-rec --label-smoothing \
  --save-dir params_mobilenetv3_large \
  --logging-file mobilenetv3_large.log
