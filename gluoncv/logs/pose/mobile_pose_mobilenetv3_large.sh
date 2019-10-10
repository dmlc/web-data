python3 train_simple_pose.py \
    --model mobile_pose_mobilenetv3_large --mode hybrid --num-joints 17 \
    --lr 0.001 --wd 0.0 --lr-mode step --lr-decay-epoch 90,120 \
    --num-epochs 140 --batch-size 32 --num-gpus 8 -j 60 \
    --dtype float32 --warmup-epochs 0 --use-pretrained-base \
    --save-dir params_mobile_pose_mobilenetv3_large \
    --logging-file mobile_pose_mobilenetv3_large.log --log-interval 100
