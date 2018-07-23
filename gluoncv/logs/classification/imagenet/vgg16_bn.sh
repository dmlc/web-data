python train_imagenet.py \
  --rec-train /media/ramdisk/rec/train.rec --rec-train-idx /media/ramdisk/rec/train.idx \
  --rec-val /media/ramdisk/rec/val.rec --rec-val-idx /media/ramdisk/rec/val.idx \
  --model vgg16_bn --mode hybrid --batch-norm \
  --batch-size 32 --num-gpus 8 -j 64 \
  --num-epochs 100 --lr 0.01 --lr-decay 0.1 --lr-decay-epoch 50,80 \
  --use-rec --save-dir params_vgg16_bn
