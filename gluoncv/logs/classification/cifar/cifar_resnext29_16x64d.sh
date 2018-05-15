python train_cifar10.py --num-epochs 300 --mode hybrid --num-gpus 4 -j 4 --batch-size 32 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 150,225 --model cifar_resnext29_16x64d
