python train_imagenet.py \
  --rec-train /media/ramdisk/rec/train.rec --rec-train-idx /media/ramdisk/rec/train.idx \
  --rec-val /media/ramdisk/rec/val.rec --rec-val-idx /media/ramdisk/rec/val.idx \
  --model resnet152_v1b --mode hybrid \
  --lr 0.4 --lr-mode cosine --num-epochs 180 --batch-size 128 --num-gpus 8 -j 60 \
  --warmup-epochs 5 --use-rec --dtype float16 --last-gamma \
  --save-dir params_resnet152_v1b_180
