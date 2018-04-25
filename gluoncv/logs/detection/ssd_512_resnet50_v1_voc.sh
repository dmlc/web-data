python train_ssd.py --gpus 0,1,2,3 -j 32 --network resnet50_v1 --data-shape 512 --dataset voc --lr 0.001 --lr-steps 160,200 --lr-decay 0.1 --epochs 240
