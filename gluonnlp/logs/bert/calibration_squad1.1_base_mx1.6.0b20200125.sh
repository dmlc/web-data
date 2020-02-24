# calibration
KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0 OMP_NUM_THREADS=1 numactl --physcpubind=0 --membind=0 python3 finetune_squad.py --only_calibration --model_parameters ./output_dir/net.params
# fp32 inference
python3 finetune_squad.py --only_predict --model_parameters ./output_dir/net.params --round_to 128 --dev_batch_size 1
# int8 inference
python3 finetune_squad.py --only_predict --model_prefix ./output_dir/model_bert_squad_quantized_customize --deploy --round_to 128 --test_batch_size 1
