python train_imagenet.py \
    --rec-train /media/ramdisk/rec/train.rec --rec-train-idx /media/ramdisk/rec/train.idx \
    --rec-val /media/ramdisk/rec/val.rec --rec-val-idx /media/ramdisk/rec/val.idx \
    --model mobilenetv2_0.75 --mode hybrid \
    --lr 0.05 --wd 0.00004 --lr-mode cosine \
    --num-epochs 150 --batch-size 64 --num-gpus 4 -j 32 \
    --label-smoothing --no-wd --warmup-epochs 5 --use-rec \
    --save-dir params_mobilenetv2_0.75 --logging-file mobilenetv2_0.75.log
