horovodrun -np 8 python3 finetune_squad.py --bert_model bert_24_1024_16 --batch_size 3 \
                         --lr 3e-5 --epochs 2 --gpu --comm_backend horovod --dtype float16
