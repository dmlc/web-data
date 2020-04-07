horovodrun -np 64 --hostfile hosts python train_horovod.py  \
--rec-train /media/ramdisk/ILSVRC2012/train.rec  \
--rec-val /media/ramdisk/ILSVRC2012/val.rec  \
--model resnest101 --lr 0.025 --num-epochs 270 --batch-size 64  \
--use-rec --dtype float32 --warmup-epochs 5 --last-gamma --no-wd \
--label-smoothing --mixup   --save-dir params_resnest101  \
--log-interval 100 --eval-frequency 5 --auto_aug --input-size 224
