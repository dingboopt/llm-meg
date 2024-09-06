#!/bin/bash

GPU_IDS="0,1,2,3,4,5,6,7"
METAROOT="/workspace/sft/yc-speech/models_hub/models/llm/Yi/Yi-6B/"
DATAROOT="./data/"
WHISPER="/workspace/sft/yc-speech/models_hub/models/ssl/whisper/large-v3"
PORT=29502

DATA_PATH=${DATAROOT}/dataset/stage2/train/dataset-20240201.json
ASR_PROMPT_PATH=${DATAROOT}/prompt/asr_prompt.txt
OUTPUT_PATH=exp/stage1-test


export CUDA_LAUNCH_BLOCKING=1
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json
# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
#export NCCL_DEBUG=INFO
#export NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_6,mlx5_7
export NCCL_SOCKET_IFNAME=ens22f0np0

export NCCL_IB_TIMEOUT=22
export NCCL_IB_GID_INDEX=3   # 走roce v2

ZERO_STAGE=0
config_json="./ds_config.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": "auto",
  "train_batch_size": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

DEEPSPEED_ARGS="--deepspeed ${config_json}"

#deepspeed --include localhost:${GPU_IDS} --master_port ${PORT} linear_pretrain.py \
#deepspeed --num_gpus 8 --num_nodes 2 --hostfile=hostfile --no_ssh_check linear_pretrain.py \
#deepspeed --hostfile=hostfile --no_ssh_check linear_pretrain.py \
torchrun --nnodes 2 --nproc_per_node 8 --master_addr "10.178.13.213" --master_port 29502 --node_rank 1 linear_pretrain.py \
    $DEEPSPEED_ARGS \
    --model_name_or_path ${METAROOT} \
    --whisper ${WHISPER} \
    --data_path ${DATA_PATH} \
    --asr_prompt_path ${ASR_PROMPT_PATH} \
    --fp16 False \
    --bf16 True \
    --freeze_llama True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --dataloader_num_workers 0 \
    --learning_rate 1e-3 \
    --warmup_steps 1000 \
    --save_steps 1000 \
    --logging_steps 25 \
    --seed 42 \
    --log_on_each_node False \
    --output_dir ${OUTPUT_PATH} \
    --max_steps 10000 \
    --greater_is_better False \
    --push_to_hub False  
