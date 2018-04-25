python train_ssd.py --gpus 0,1,2,3 -j 32 --batch-size 32 --network vgg16_atrous --data-shape 512 --dataset voc --lr 0.001 --lr-steps 160,200 --lr-decay 0.1 --epochs 240
