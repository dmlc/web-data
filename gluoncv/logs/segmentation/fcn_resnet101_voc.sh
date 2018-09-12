# First finetuning COCO dataset pretrained model on augmented set
# If you would like to train from scratch on COCO, please see fcn_resnet101_coco.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset pascal_aug --model-zoo fcn_resnet101_coco --lr 0.001 --syncbn --ngpus 4 --checkname res101
# Finetuning on original set
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset pascal_voc --model fcn --aux --backbone resnet101 --lr 0.0001 --syncbn --ngpus 4 --checkname res101 --resume runs/pascal_aug/fcn/res101/checkpoint.params
