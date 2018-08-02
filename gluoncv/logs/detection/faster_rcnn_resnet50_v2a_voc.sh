python3 train_faster_rcnn.py --gpus 0,1,2,3 --network resnet50_v2a -j 8 --dataset voc --lr 0.001 --lr-decay-epoch 14,20 --lr-decay 0.1 --epochs 30
