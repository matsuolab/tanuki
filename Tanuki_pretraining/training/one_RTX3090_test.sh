#!/bin/bash

#conda activate 
source ./miniconda3/etc/profile.d/conda.sh
conda activate tanuki_pretraining_py310_TEv1.7_FAv2.5.7

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0

# hostfile
export NUM_GPU_PER_NODE=1

# model config
# llama-3-8b: https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/config.json
#HIDDEN_SIZE=4096
#FFN_HIDDEN_SIZE=14336 # intermediate size (HuggingFace)
#NUM_LAYERS=32
#NUM_HEADS=32
#NUM_KEY_VALUE_HEADS=8

model_size=7
NUM_LAYERS=8

HIDDEN_SIZE=256
FFN_HIDDEN_SIZE=1024 # hiddensizeの3.5倍
#ffn_hidden_size: 24576 # hiddensizeの4倍
#num_attn_heads: 48
NUM_HEADS=4
NUM_KEY_VALUE_HEADS=4

SEQ_LENGTH=2048

# distributed settings
TENSOR_PARALLEL_SIZE=1  # fixed
PIPELINE_PARALLEL_SIZE=1 # num layers 32: Llama-2 8B
CONTEXT_PARALLEL_SIZE=1

# training config
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8
TRAIN_STEPS=175000
LR_DECAY_ITERS=20

LR=1.0E-5
MIN_LR=1.0E-6
LR_WARMUP_STEPS=10
WEIGHT_DECAY=0.1
GRAD_CLIP=1

# model config
TOKENIZER_MODEL="./training/tokernizer/tokenizer_scale200.model"
CHECKPOINT_LOAD_DIR="./training/checkpoints/Llama-3-8b/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}"
CHECKPOINT_SAVE_DIR="./training/checkpoints/Llama-3-8/tp${TENSOR_PARALLEL_SIZE}-pp${PIPELINE_PARALLEL_SIZE}-ct${CONTEXT_PARALLEL_SIZE}-LR${LR}-MINLR${MIN_LR}-WD${WEIGHT_DECAY}-WARMUP${LR_WARMUP_STEPS}"

log_path="${CHECKPOINT_SAVE_DIR}/log"

DATA_PATH_LIST=(
"./training/training_corpus/hanrei_1/combined_part_01_to_08_text_sentence"
"./training/training_corpus/hanrei_2/combined_part_18_to_25_text_sentence"
#"add more"
)

for path in "${DATA_PATH_LIST[@]}"; do
    cache_dir="$path/cache"
    if [ -d "$cache_dir" ]; then
        echo "Deleting cache directory: $cache_dir"
        rm -rf "$cache_dir"
    else
        echo "Cache directory not found: $cache_dir"
    fi
done

# 配列をスペース区切りの文字列に変換
TRAIN_DATA_PATH=$(IFS=' '; echo "${DATA_PATH_LIST[*]}")

echo $TRAIN_DATA_PATH

mkdir -p ${CHECKPOINT_SAVE_DIR}
mkdir -p ${log_path}

# checkpoint load
if [[ -f "${CHECKPOINT_LOAD_DIR}/latest_checkpointed_iteration.txt" ]]; then
  # resume training
  CHECKPOINT_ARGS="--load ${CHECKPOINT_LOAD_DIR}"
else
  CHECKPOINT_ARGS="--load ${CHECKPOINT_LOAD_DIR} --no-load-rng --no-load-optim"
fi

JOB_NAME="Llama-3_test"

# run
megatron_options="  \
  --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --context-parallel-size ${CONTEXT_PARALLEL_SIZE} \
  --sequence-parallel \
  --use-distributed-optimizer \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
  --num-attention-heads ${NUM_HEADS} \
  --group-query-attention \
  --num-query-groups ${NUM_KEY_VALUE_HEADS} \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
	--tokenizer-type SentencePieceTokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  ${CHECKPOINT_ARGS} \
  --save ${CHECKPOINT_SAVE_DIR} \
  --data-path ${TRAIN_DATA_PATH} \
  --split 998,1,1 \
  --distributed-backend nccl \
  --init-method-std 0.008 \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style cosine \
  --lr-decay-iters ${LR_DECAY_ITERS} \
  --weight-decay ${WEIGHT_DECAY} \
  --clip-grad ${GRAD_CLIP} \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-05 \
  --log-interval 1 \
  --save-interval 500 \
  --eval-interval 20 \
  --eval-iters 10 \
  --bf16 \
  --use-checkpoint-args \
  --untie-embeddings-and-output-weights \
  --no-position-embedding \
  --position-embedding-type rope \
  --rope-theta 500000.0 \
  --disable-bias-linear \
  --use-mcore-models \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --no-masked-softmax-fusion \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --swiglu \
  --use-flash-attn \
  --attention-softmax-in-fp32 \
  --recompute-activations \
  --recompute-granularity "selective" \
  --transformer-impl "transformer_engine" \
  --use-z-loss \
  --log-throughput \
	"

  # --bf16 \
  # --transformer-impl "transformer_engine" \
  # --fp8-format 'hybrid' \
  # --fp8-amax-compute-algo max \
  # --fp8-amax-history-len 1024 \
  # --wandb-name ${JOB_NAME} \
  # --wandb-project "Llama-3-1B" \
  # --wandb-entity "weblab-geniac1" \
  # --with-packing \

current_time=$(date "+%Y.%m.%d_%H.%M.%S")

log_file="${log_path}/llma3_31_${node_rank}_${current_time}.log"

# NNODES=1
# NODE_RANK=0
# MASTER_ADDR=n_a
# MASTER_PORT=65400
# DISTRIBUTED_ARGS="--nproc_per_node $NUM_GPU_PER_NODE --nnodes $NNODES  --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# echo "dist args"
# echo $DISTRIBUTED_ARGS
# echo ""

run_cmd="torchrun ./Megatron-LM/pretrain_gpt.py
 ${megatron_options} \
 2>&1 | tee ${log_file}"

echo ${run_cmd}
eval ${run_cmd}
set +x