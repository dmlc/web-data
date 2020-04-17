# calibration
KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0 OMP_NUM_THREADS=1 numactl --physcpubind=0 --membind=0 python3 finetune_classifier.py --task_name MRPC --only_calibration --model_parameters ./output_dir/model_bert_MRPC_4.params
# fp32 inference
python3 finetune_classifier.py --task_name MRPC --epoch 1 --only_inference --model_parameters ./output_dir/model_bert_MRPC_4.params --round_to 128 --dev_batch_size 1
# int8 inference
python3 finetune_classifier.py --task_name MRPC --epoch 1 --only_inference --model_prefix ./output_dir/model_bert_MRPC_quantized_customize --deploy --round_to 128 --dev_batch_size 1
