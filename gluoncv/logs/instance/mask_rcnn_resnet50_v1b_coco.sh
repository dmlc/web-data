python3 ../gluon-cv/scripts/instance/mask_rcnn/train_mask_rcnn.py --gpus 0,1,2,3,4,5,6,7
python ../gluon-cv/scripts/instance/mask_rcnn/eval_mask_rcnn.py --gpus 0,1,2,3,4,5,6,7 --pretrained mask_rcnn_resnet50_v1b_coco_best.params &>> mask_rcnn_resnet50_v1b_coco_train.log
