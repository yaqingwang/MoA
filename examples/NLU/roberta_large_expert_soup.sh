export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
task_name=qqp
model=roberta-large
path=/home/v-yaqingwang/azure_storage/projects/Efficient_tuning/roberta_large_cola_soup/pt-results/Efficient_tuning-cf6f4f93-roberta_large_cola_soup-search_cola_roberta-large_seed_2_80_inference_3_16_4_4_consist_1_0_1_3e-4-2554e7e3
path=/home/v-yaqingwang/azure_storage/projects/Efficient_tuning/roberta_large_mnli_soup_sharing_up/pt-results/Efficient_tuning-cf6f4f93-roberta_large_mnli_soup_sharing_up-search_mnli_roberta-large_seed_0_20_inference_3_16_4_4_consist_1_0_1_3e-4-e2087777
path=/home/v-yaqingwang/azure_storage/projects/Efficient_tuning/roberta_large_mrpc_soup/pt-results/Efficient_tuning-cf6f4f93-roberta_large_mrpc_soup-search_mrpc_roberta-large_seed_0_60_inference_3_16_4_4_consist_1_0_1_3e-4-8d645003
path=/home/v-yaqingwang/azure_storage/projects/Efficient_tuning/roberta_large_sst2_soup_sharing_up/pt-results/Efficient_tuning-cf6f4f93-roberta_large_sst2_soup_sharing_up-search_sst2_roberta-large_seed_0_20_inference_3_16_4_4_consist_1_0_1_3e-4-daa46e09
path=/home/v-yaqingwang/azure_storage/projects/Efficient_tuning/roberta_large_stsb_soup_sharing_up/pt-results/Efficient_tuning-cf6f4f93-roberta_large_stsb_soup_sharing_up-search_stsb_roberta-large_seed_0_80_inference_3_16_4_4_consist_0_0_1_3e-4-67bb626a
path=/home/v-yaqingwang/azure_storage/projects/Efficient_tuning/roberta_large_qnli_soup_sharing_up/pt-results/Efficient_tuning-cf6f4f93-roberta_large_qnli_soup_sharing_up-search_qnli_roberta-large_seed_0_20_inference_3_16_4_4_consist_1_0_1_3e-4-300403c5
path=/home/v-yaqingwang/azure_storage/projects/Efficient_tuning/roberta_large_rte_soup_sharing_up_consistent/pt-results/Efficient_tuning-cf6f4f93-roberta_large_rte_soup_sharing_up_consistent-search_rte_roberta-large_seed_0_60_inference_3_16_4_4_consist_1_0_1_5e-4-57cffd92
path=/home/v-yaqingwang/azure_storage/projects/Efficient_tuning/roberta_large_qqp_soup_sharing_up/pt-results/Efficient_tuning-cf6f4f93-roberta_large_qqp_soup_sharing_up-search_qqp_roberta-large_seed_0_80_inference_3_16_4_4_consist_1_0_1-69b1d2a1
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
--adapter_size 16 \
--num_experts 4 \
--seed 0 \
--inference_level 3 \
--load_best_model_at_end \
--metric_for_best_model "accuracy" \
--sharing_up 1 \

cp ${path}/*  $output_dir