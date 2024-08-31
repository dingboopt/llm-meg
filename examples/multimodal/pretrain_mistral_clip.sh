#!/bin/bash
# Pretrain a multimodal model.
set -x

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
MODEL_NAME="mcore-llava-mistral-7b-instruct-clip336-pretraining"

# Check that the user has set an output path for model checkpoints.
if [[ -z $WORKSPACE ]]; then
    echo "Please set WORKSPACE for storing your model checkpoints."
    exit 1
fi

DATE=`date '+%Y-%m-%d'`                                                                                                                                                                                      
HOUR=`date '+%H-%M-%S'`


SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
#OUTPUT_BASE="${WORKSPACE}/output${DATE}_${HOUR}"

OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

FINETUNE_DIR=${OUTPUT}/checkpoints
LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"

if [[ -z $LOAD_NAME ]]; then
    echo "Please set LOAD_NAME for input model name."
    exit 1
fi

if [[ -z $TOKENIZER_MODEL ]]; then
    echo "Please set TOKENIZER_MODEL for tokenizer model name."
    exit 1
fi

CHECKPOINT_DIR="${WORKSPACE}/${LOAD_NAME}/mistral_instruct_clip336_tp4_combined_mcore"

DATA_TRAIN="${SOURCE}/examples/multimodal/pretrain_dataset.yaml"
DATA_VALID="${SOURCE}/examples/multimodal/pretrain_dataset.yaml"

DEBUG=0
if [[ $DEBUG -eq 1 ]]; then
    BZ=32
    NW=2
    HD=0.0
    LI=1
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
else
    BZ=256
    NW=2
    HD=0.1
    LI=1
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
fi

OPTIONS=" \
    --apply-layernorm-1p \
    --attention-softmax-in-fp32 \
    --use-checkpoint-args \
    --use-distributed-optimizer \
    --transformer-impl transformer_engine \
    --use-te \
    --normalization RMSNorm \
    --group-query-attention \
    --num-query-groups 8 \
    --no-masked-softmax-fusion \
    --num-workers ${NW} \
    --use-flash-attn \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --swiglu \
    --attention-dropout 0.0 \
    --hidden-dropout ${HD} \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 4096 \
    --ffn-hidden-size 14336 \
    --train-iters 20000 \
    --micro-batch-size 4 \
    --global-batch-size ${BZ} \
    --lr-decay-iters 20000 \
    --lr-warmup-fraction .01 \
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --log-interval ${LI} \
    --eval-iters 10 \
    --eval-interval 1000 \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${WORKSPACE}/${TOKENIZER_MODEL} \
    --data-path ${DATA_TRAIN} \
    --valid-path ${DATA_VALID} \
    --prompt-path ${SOURCE}/examples/multimodal/manual_prompts.json \
    --save-interval 1000 \
    --save ${FINETUNE_DIR} \
    --load ${FINETUNE_DIR} \
    --dataloader-save ${FINETUNE_DIR}/dataloader \
    --pretrained-checkpoint ${CHECKPOINT_DIR} \
    --split 100,0,0 \
    --clip-grad 1.0 \
    --weight-decay 1e-2 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --bf16 \
    --eod-mask-loss \
    --freeze-LM \
    --freeze-ViT \
    --patch-dim 14 \
    --img-h 336 \
    --img-w 336 \
    --dataloader-type external \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --language-model-type=mistral_7b \
    --disable-vision-class-token \
    ${EXTRA_ARGS} \
    --distributed-timeout-minutes 60 \
    --allow-missing-vision-projection-checkpoint \
	--ckpt-format torch \
	

"
# dump config and in/out tensor
#--config-logger-dir ${OUTPUT}/config/ \
export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NONDETERMINISTIC_ATTN}

GPUS_PER_NODE=8
mkdir -p ${OUTPUT}/logs/$NODE_RANK

if [ ! -z $DS ]; then
	echo "************. use deepspeed ********************."
	deepspeed --hostfile=hostfile  --no_ssh_check --master_port 60410  examples/multimodal/train.py ${OPTIONS} 2>&1| tee ${OUTPUT}/logs/log

elif [ -z $NUM_NODES ] || [ "$NUM_NODES" -le 1 ]; then
	echo "One node!!!!!!!!!."
	torchrun --nproc_per_node $GPUS_PER_NODE examples/multimodal/train.py ${OPTIONS} 2>&1| tee ${OUTPUT}/logs/log
else
	echo "multi nodes   ######."
	export CUDA_DEVICE_MAX_CONNECTIONS=1
	export GLOO_SOCKET_IFNAME=ens22f0np0

	# Change for multinode config
	MASTER_ADDR=$MASTER_ADDR
	MASTER_PORT=60235
	#NUM_NODES=2
	NODE_RANK=$NODE_RANK
	WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))


	DISTRIBUTED_ARGS=(
	    --nproc_per_node $GPUS_PER_NODE 
	    --nnodes $NUM_NODES 
	    --master_addr $MASTER_ADDR 
		--node-rank $NODE_RANK
	    --master_port $MASTER_PORT
	)



	torchrun ${DISTRIBUTED_ARGS[@]} examples/multimodal/train.py ${OPTIONS} 2>&1| tee ${OUTPUT}/logs/$NODE_RANK/log
fi

