horovodrun -np 64 --hostfile hosts python train_horovod.py  \
--rec-train /media/ramdisk/ILSVRC2012/train.rec  \
--rec-val /media/ramdisk/ILSVRC2012/val.rec  \
--model resnest50 --lr 0.05 --num-epochs 270 --batch-size 128  \
--use-rec --dtype float32 --warmup-epochs 5 --last-gamma --no-wd \
--label-smoothing --mixup   --save-dir params_resnest50  \
--log-interval 100 --eval-frequency 5 --auto_aug --input-size 224
