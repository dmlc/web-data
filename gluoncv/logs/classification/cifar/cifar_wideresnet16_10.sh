python train_cifar10.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 8 --batch-size 128 --wd 0.0005 --lr 0.1 --lr-decay 0.2 --lr-decay-epoch 60,120,160 --model cifar_wideresnet16_10
