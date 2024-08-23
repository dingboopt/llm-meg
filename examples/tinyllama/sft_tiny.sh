#!/bin/bash
# This example script is contributed by external user https://github.com/nrailgun
set -x
MEGATRON_PATH=../../
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

DATE=`date '+%Y-%m-%d'`
HOUR=`date '+%H-%M-%S'`
TAG="$1"

# 116486373 samples/epoche; 2048 seq/sample 
DATASET=/cloudfs-data/db/data/the_pile_deduplicated_tinyllama_tokenized_idx/pie_total

CHECKPOINT_PATH=/cloudfs-data/db/model/TinyLlama_v1.1/
SAVE_CHECKPOINT_PATH=$TAG
TOKENIZER_PATH=${CHECKPOINT_PATH}
TENSORBOARD_PATH=output_dir/tensorboard/$DATE-llama-13b$TAG


MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=1024 # e.g. llama: 4M tokens
TRAIN_STEPS=341268 # 116486373 * 3 /1024

# parallel parameter
TP=1
PP=1
# zero 1
DO=true
# sequence parallel
SP=true
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

NUM_LAYERS=22
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=5632

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups 4"


SEQ_LENGTH=2048






#CUSTOM_ARGS=" \
#    --no-load-optim \
#    --finetune \
#    "

# learning rate and optimizer hyper para 
#train-iter = total_sample / gbs
#lr_decay =  train-iter * 0.95
#warmup = train-iter*0.03
LR_DECAY_ITERS=340268
LR_WARMUP_ITERS=1000

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.99 \
    --adam-eps 1e-8 \
    --lr 4e-4 \
    --min-lr 4e-4 \
    --lr-decay-style cosine \
    --lr-decay-iters $LR_DECAY_ITERS \
    --lr-warmup-iters $LR_WARMUP_ITERS \
    --clip-grad 1.0 \
    --weight-decay 0. \
    "


deepspeed --hostfile=hostfile  --no_ssh_check --master_port 60410 \
	pretrain_llama.py \
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
	   --data-path ${DATASET} 100 \
	   --split 98,1,1 \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $INTERMEDIATE_SIZE \
       --num-attention-heads $NUM_ATTN_HEADS \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_STEPS \
       --save $SAVE_CHECKPOINT_PATH \
	   --load $SAVE_CHECKPOINT_PATH \
	   --tokenizer-model $CHECKPOINT_PATH \
       --tokenizer-type HuggingFaceTokenizer \
	   --make-vocab-size-divisible-by 4 \
       --distributed-backend nccl \
       --log-interval 1 \
       --save-interval 1000 \
       --eval-interval 1000 \
       --eval-iters 10 \
   	   --seed 42 \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization RMSNorm \
       --disable-bias-linear \
        --no-rope-fusion \
	   --tensorboard-dir $TENSORBOARD_PATH \
		--tensorboard-queue-size 5 \
		--log-timers-to-tensorboard \
		--ckpt-format torch \
		--log-validation-ppl-to-tensorboard \
      2>&1 |tee ${TAG}_${DATE}_${HOUR}.log
