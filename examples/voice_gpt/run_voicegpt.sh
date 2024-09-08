#!/bin/bash
# This example script is contributed by external user https://github.com/nrailgun
set -x
MEGATRON_PATH=../../
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

DATE=`date '+%Y-%m-%d'`
HOUR=`date '+%H-%M-%S'`
TAG="$1"

DATASET=/cloudfs/db/Megatron-LM/examples/voice_gpt/pretrain_dataset.yaml

CHECKPOINT_PATH=/cloudfs/db/Megatron-LM/examples/voice_gpt/test
SAVE_CHECKPOINT_PATH=${TAG}
TOKENIZER_PATH=${CHECKPOINT_PATH}
TENSORBOARD_PATH=output_dir/tensorboard/$DATE-llama-13b$TAG


MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=64 # e.g. llama: 4M tokens
TRAIN_STEPS=2 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps

# parallel parameter
TP=4
PP=1
# zero 1
DO=true
# sequence parallel
SP=False
# full /sel /none
AC=sel
# flash attention
FL=true
# transformer engine
TE=true
# precision fp16/bf16/fp8
PR=bf16


if [ $AC = full ]; then
    activation_checkpoint_options=" \
        --recompute-num-layers 1 \
		    --recompute-method uniform \
		    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
                    "
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16"
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16 \
        --fp8-hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024 \
        --transformer-impl transformer_engine"
fi

if [ $DO = true ]; then
    do_options=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

if [ $FL = true ]; then
    flash_options=" \
		    --use-flash-attn"

elif [ $FL = false ]; then
    flash_options=" \
                    "
fi

if [ $TE = true ]; then
    te_options=" \
		    --transformer-impl transformer_engine"

elif [ $TE = false ]; then
    te_options=" \
                    "
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi






# model parameter

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=11008

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups 4"


SEQ_LENGTH=2048






CUSTOM_ARGS=" \
    --no-load-optim \
    --finetune \
    "

# learning rate and optimizer hyper para 
#train-iter = total_sample / gbs
#lr_decay =  train-iter * 0.95
#warmup = train-iter*0.03
LR_DECAY_ITERS=9450
LR_WARMUP_ITERS=1050

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.99 \
    --adam-eps 1e-8 \
    --lr 1e-3 \
    --min-lr 1e-6 \
    --lr-decay-style cosine \
    --lr-decay-iters $LR_DECAY_ITERS \
    --lr-warmup-iters $LR_WARMUP_ITERS \
    --clip-grad 1.0 \
    --weight-decay 0. \
    "


deepspeed --include=localhost  --no_ssh_check --master_port 60411 \
	train.py \
	   $CUSTOM_ARGS \
	   $OPTIMIZER_ARGS \
	   ${activation_checkpoint_options} \
	   ${pr_options} \
	   ${load_options} \
	   ${te_options} \
	   ${activation_checkpoint_options} \
	   ${do_options} ${flash_options} \
	   ${sp_options} \
	   ${gqa_options} \
	   --data-path ${DATASET} \
	   --valid-path ${DATASET} \
	   --split 1000,0,0 \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $INTERMEDIATE_SIZE \
       --num-attention-heads $NUM_ATTN_HEADS \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings 4096 \
       --train-iters $TRAIN_STEPS \
       --save $SAVE_CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
	   --tokenizer-model $CHECKPOINT_PATH \
       --tokenizer-type HuggingFaceTokenizer \
	   --make-vocab-size-divisible-by 4 \
       --distributed-backend nccl \
       --log-interval 1 \
       --save-interval 500 \
	   --ckpt-format torch \
       --eval-interval 1000 \
       --eval-iters 10 \
   	   --seed 42 \
       --attention-dropout 0 \
       --hidden-dropout 0 \
	   --rotary-base 5000000 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
	   --distributed-timeout-minutes 1 \
       --swiglu \
       --normalization RMSNorm \
       --disable-bias-linear \
        --use-mcore-models \
        --no-rope-fusion \
	   --tensorboard-dir $TENSORBOARD_PATH \
		--tensorboard-queue-size 5 \
		--log-timers-to-tensorboard \
		--dataloader-type external \
		--encoder-path /cloudfs-data/db/model/large-v3/ \
		--log-validation-ppl-to-tensorboard \
		--adapter-list  affine:1280 linear+mlp:1280 \
      2>&1 |tee sft_llama2$TAG.log
