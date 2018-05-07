python train_cifar10.py --num-epochs 200 --mode hybrid --num-gpus 1 -j 2 --batch-size 128 --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 100,150 --model cifar_resnet56_v1
