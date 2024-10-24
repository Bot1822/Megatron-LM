#!/bin/bash

set -ex
_THIS_DIR=$(dirname "$0")

# Runs the "175B" parameter model
# memory 46.2-50.3G
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONPATH=/workspace/dump_data_tool:$PYTHONPATH
# export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

TENSORBOARD_LOGS_PATH=./tensorboard
# DATA_PATH=/mnt/host-path/nfs-data3/datasets/xlc/llama3-8b/wudao-dataset/wudao_llama3bpe_content_document
# TOKENIZER_MODEL=/ceph/User/E01442/workspace-hgct-70b/megatron-lm/examples/llama2/tokenizer


DATA_PARENT_PATH=${DATA_PARENT_PATH:-/share/teamFPA/llama2-series/70b_loss_acc_compare/RedPajama-Data-1T-Sample}
DATA_PATH="$DATA_PARENT_PATH/redpajama-llama2_text_document"
VOCAB_FILE=$_THIS_DIR/tokenizer
TOKENIZER_MODEL=$_THIS_DIR/tokenizer/tokenizer.model

TP=4
PP=2
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=16
NHIDDEN=8192
FFN_HIDDEN_SIZE=28672
NLAYERS=8
NHEADS=64
NKVHEADS=8 
SEQ_LEN=4096
STEP=10

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers ${NLAYERS} 
    --hidden-size ${NHIDDEN} 
    --ffn-hidden-size ${FFN_HIDDEN_SIZE} 
    --num-attention-heads ${NHEADS} 
    --group-query-attention
    --num-query-groups ${NKVHEADS} 
    --seq-length ${SEQ_LEN}
    --max-position-embeddings 4096 
    --normalization RMSNorm
    --swiglu
    --attention-softmax-in-fp32
    --disable-bias-linear 
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --position-embedding-type rope
    --rotary-base 500000
    --use-cpu-initialization
    --untie-embeddings-and-output-weights
    --no-create-attention-mask-in-dataloader
    --initial-loss-scale 65536 
    --cross-entropy-loss-fusion 
    --vocab-size 32000
)
    # --recompute-num-layers 1 
    # --recompute-granularity=full 
    # --recompute-method=uniform 
    ## --make-vocab-size-divisible-by 128

TRAINING_ARGS=(
    --micro-batch-size ${MICRO_BATCH_SIZE} 
    --global-batch-size ${GLOBAL_BATCH_SIZE} 
    --train-iters ${STEP}
    --lr 1e-5
    --min-lr 1e-6
    --lr-decay-iters 300
    --lr-decay-style cosine
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.008
    --clip-grad 1.0
    --bf16
    --transformer-impl transformer_engine
    --use-distributed-optimizer
    --use-flash-attn
    --sequence-parallel
)
    # --use-mcore-models

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP} 
    --pipeline-model-parallel-size ${PP} 
)

DATA_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --data-path $DATA_PATH 
    --split 100,0,0
)

DP=$((WORLD_SIZE / TP / PP))
ALL_LOGS_PATH=logs/
mkdir -p ${ALL_LOGS_PATH}
LOGS_PATH=mb${MICRO_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}_l${NLAYERS}_tp${TP}_pp${PP}_dp${DP}_n${NNODES}
mkdir -p $ALL_LOGS_PATH/${LOGS_PATH}
timestamp=$(date +%s)
TENSORBOARD_LOGS_PATH=$ALL_LOGS_PATH/${LOGS_PATH}/${timestamp}

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000
    --eval-interval 1000
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    2>&1 | tee -a $ALL_LOGS_PATH/${LOGS_PATH}/${timestamp}.log
