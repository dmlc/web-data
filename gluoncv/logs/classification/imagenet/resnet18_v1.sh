python train_imagenet.py --batch-size 64 --num-gpus 4 -j 32 --mode hybrid --num-epochs 150 --lr 0.1 --momentum 0.9 --wd 0.0001 --lr-decay 0.1 --lr-decay-epoch 30,60,90 --model resnet18_v1
