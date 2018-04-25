python train_cifar10.py --num-epochs 240 --mode hybrid --num-gpus 1 -j 8 --batch-size 128 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 80,160 --model cifar_resnet56_v1
