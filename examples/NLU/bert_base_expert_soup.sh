export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
task_name=qqp
model=bert-base-uncased
path=/home/v-yaqingwang/azure_storage/projects/Efficient_tuning/bert_base_cola_moreepochs/pt-results/Efficient_tuning-cf6f4f93-bert_base_cola_moreepochs-search_cola_bert-base-uncased_seed_0_100_inference_3_48_4_1_consist_1_0_1_5e-4-af159e94
path=/home/v-yaqingwang/azure_storage/projects/Efficient_tuning/bert_base_mrpc_rrandomtry/pt-results/Efficient_tuning-cf6f4f93-bert_base_mrpc_rrandomtry-search_mrpc_bert-base-uncased_seed_0_100_inference_3_48_4_1_consist_1_0_1_4e-4-01608ce0
path=/home/v-yaqingwang/azure_storage/projects/Efficient_tuning/bert_base_sst2/pt-results/Efficient_tuning-cf6f4f93-bert_base_sst2-search_sst2_bert-base-uncased_seed_0_40_inference_3_48_4_4_consist_1_0_1_4e-4-938769d3
path=/home/v-yaqingwang/azure_storage/projects/Efficient_tuning/bert_base_stsb_moreepochs/pt-results/Efficient_tuning-cf6f4f93-bert_base_stsb_moreepochs-search_stsb_bert-base-uncased_seed_0_80_inference_3_48_4_2_consist_1_0_1_5e-4-4280f147
path=/home/v-yaqingwang/azure_storage/projects/Efficient_tuning/bert_base_qnli/pt-results/Efficient_tuning-cf6f4f93-bert_base_qnli-search_qnli_bert-base-uncased_seed_0_20_inference_3_48_4_4_consist_1_0_1_4e-4-14ded7bd
path=/home/v-yaqingwang/azure_storage/projects/Efficient_tuning/bert_base_mnli_soup/pt-results/Efficient_tuning-cf6f4f93-bert_base_mnli_soup-search_mnli_bert-base-uncased_seed_0_40_inference_3_48_4_4_consist_1_0_1_4e-4-768fc889
path=/home/v-yaqingwang/azure_storage/projects/Efficient_tuning/bert_base_rte_moreepochs/pt-results/Efficient_tuning-cf6f4f93-bert_base_rte_moreepochs-search_rte_bert-base-uncased_seed_0_80_inference_3_48_4_2_consist_1_0_1_5e-4-df10b086
path=/home/v-yaqingwang/azure_storage/projects/Efficient_tuning/bert_base_qqp_soup_all/pt-results/Efficient_tuning-cf6f4f93-bert_base_qqp_soup_all-search_qqp_bert-base-uncased_seed_0_60_inference_3_48_4_4_consist_1_0_1_5e-4-7761f1b2/checkpoint-301358
export output_dir="/home/v-yaqingwang/Projects/models/${model}/${task_name}"
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path $model \
--task_name $task_name \
--expert_soup_path ${path}/pytorch_model.bin \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 32 \
--learning_rate 5e-4 \
--num_train_epochs 100 \
--output_dir $output_dir/model \
--overwrite_output_dir \
--logging_steps 100 \
--logging_dir $output_dir/log \
--evaluation_strategy epoch \
--save_strategy epoch \
--warmup_ratio 0.06 \
--apply_expert_soup \
--adapter_type houlsby \
--adapter_size 48 \
--num_experts 4 \
--seed 0 \
--inference_level 3 \
--load_best_model_at_end \
--metric_for_best_model "accuracy" \
--sharing_up 1 \

cp ${path}/*  $output_dir