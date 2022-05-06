

## Adapting to the GLUE Benchmark
Our experiments on the GLUE benchmark are run on 4 NVIDIA Tesla V100 GPU cards out of a DGX-1. The results may vary due to different GPU models, drivers, CUDA SDK versions, floating-point precisions, and random seeds. 
We report below the dev set results, taking the medium over 5 runs:



## Download LoRA checkpoints

|   | Dataset  | BERT base 110M <br>   | RoBERTa large 355M <br>  |
|---|----------|--------------------|----------------------|
|   | MNLI     |[8.5 MB](https://github.com/yaqingwang/MoA/releases/download/bert_base/pytorch_model_mnli_expert_soup.bin) |[11.7 MB](https://github.com/yaqingwang/MoA/releases/download/roberta_large/pytorch_model_mnli_expert_soup.bin) |
|   | SST2     |[8.5 MB](https://github.com/yaqingwang/MoA/releases/download/bert_base/pytorch_model_sst2_expert_soup.bin)  |[11.7 MB](https://github.com/yaqingwang/MoA/releases/download/roberta_large/pytorch_model_sst2_expert_soup.bin)  |
|   | MRPC     |[8.5 MB](https://github.com/yaqingwang/MoA/releases/download/bert_base/pytorch_model_mrpc_expert_soup.bin)  |[11.7 MB](https://github.com/yaqingwang/MoA/releases/download/roberta_large/pytorch_model_mrpc_expert_soup.bin)  |
|   | CoLA     |[8.5 MB](https://github.com/yaqingwang/MoA/releases/download/bert_base/pytorch_model_cola_expert_soup.bin)  |[11.7 MB](https://github.com/yaqingwang/MoA/releases/download/roberta_large/pytorch_model_cola_expert_soup.bin)  |
|   | QNLI     |[8.5 MB](https://github.com/yaqingwang/MoA/releases/download/bert_base/pytorch_model_qnli_expert_soup.bin)  |[11.7 MB](https://github.com/yaqingwang/MoA/releases/download/roberta_large/pytorch_model_qnli_expert_soup.bin)  |
|   | QQP      |[8.5 MB](https://github.com/yaqingwang/MoA/releases/download/bert_base/pytorch_model_qqp_expert_soup.bin)  |[11.7 MB](https://github.com/yaqingwang/MoA/releases/download/roberta_large/pytorch_model_qqp_expert_soup.bin)  |
|   | RTE      |[8.5 MB](https://github.com/yaqingwang/MoA/releases/download/bert_base/pytorch_model_rte_expert_soup.bin)  |[11.7 MB](https://github.com/yaqingwang/MoA/releases/download/roberta_large/pytorch_model_rte_expert_soup.bin)  |
|   | STSB     |[8.5 MB](https://github.com/yaqingwang/MoA/releases/download/bert_base/pytorch_model_stsb_expert_soup.bin)  |[11.7 MB](https://github.com/yaqingwang/MoA/releases/download/roberta_large/pytorch_model_stsb_expert_soup.bin)  |

## Steps to reproduce our results
### Create and activate conda env
```console
conda env create -f environment.yml
```
### Install the pre-requisites
lora:
```console
pip install -e ..
```
NLU:
```console
pip install -e .
```

We also provide the shell scripts for roberta-base and roberta-large ( {roberta_large|roberta_base}_{task name}.sh ).

### Evaluate the checkpoints
```console
export num_gpus=1
export CUBLAS_WORKSPACE_CONFIG=":16:8" # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export PYTHONHASHSEED=0
task_name=qqp
model=roberta-large
export output_dir="./models/${model}/${task_name}"
python -m torch.distributed.launch --nproc_per_node=$num_gpus \
examples/text-classification/run_glue.py \
--model_name_or_path $model \
--task_name $task_name \
--do_train \
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
--sharing_up 1 \

```

