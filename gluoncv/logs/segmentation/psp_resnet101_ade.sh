CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset ade20k --model psp --aux --backbone resnet101 --lr 0.01 --syncbn --ngpus 4 --checkname res101
