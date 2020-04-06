python finetune_squad.py --bert_model bert_24_1024_16 --optimizer adam --accumulate 8 \
                         --batch_size 4 --lr 3e-5 --epochs 2 --gpu --null_score_diff_threshold -2.0 \
                         --version_2
