python train_imagenet.py \
  --rec-train /media/ramdisk/rec/train.rec --rec-train-idx /media/ramdisk/rec/train.idx \
  --rec-val /media/ramdisk/rec/val.rec --rec-val-idx /media/ramdisk/rec/val.idx \
  --model resnet34_v1b --mode hybrid \
  --lr 0.4 --lr-mode cosine --num-epochs 120 --batch-size 256 --num-gpus 4 -j 30 \
  --warmup-epochs 5 --use-rec --last-gamma \
  --save-dir params_resnet34_v1b
