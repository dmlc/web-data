python3 ../gluon-cv/scripts/classification/imagenet/train_imagenet.py \
  --rec-train /mnt/ramdisk/rec/train.rec --rec-train-idx /mnt/ramdisk/rec/train.idx \
  --rec-val /mnt/ramdisk/rec/val.rec --rec-val-idx /mnt/ramdisk/rec/val.idx \
  --model darknet53 --mode hybrid \
  --lr 0.4 --lr-mode cosine --num-epochs 180 --batch-size 128 --num-gpus 8 -j 60 \
  --warmup-epochs 5 --use-rec --dtype float16 \
  --save-dir darknet53
