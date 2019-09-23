python finetune_classifier.py --log_interval 100 --task_name MNLI --bert_model roberta_12_768_12 \
       --gpu 0 --epochs 10 --dtype float32 --early_stop 2 --lr 1e-5 --batch_size 32 \
       --bert_dataset openwebtext_ccnews_stories_books_cased --warmup 0.06
