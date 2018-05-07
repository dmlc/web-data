python train_mixup_cifar10.py --num-epochs 220 --mode hybrid --num-gpus 2 -j 2 --batch-size 64 --wd 0.0005 --lr 0.1 --lr-decay 0.2 --lr-decay-epoch 60,120,160 --model cifar_wideresnet28_10
