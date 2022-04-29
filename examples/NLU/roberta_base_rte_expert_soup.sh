export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
export output_dir="./rte"
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path roberta-base \
--task_name rte \
--do_train \
--do_eval \
--max_seq_length 256 \
--per_device_train_batch_size 32 \
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
--metric_for_best_model "accuracy"
