#!/bin/sh

# source /storage5/shared/jk/miniconda3/etc/profile.d/conda.sh
# conda activate share-jk_py310_TEv1.7_FAv2.5.7

# distributed settings
TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=4

# model config
HF_CHECKPOINT_DIR=/home/knishizawa/GENIAC/defactoring/Tanuki_pretraining/training/HF_data/Tanuki-8B-base-v1.0
MEGATRON_CHECKPOINT_DIR=/home/knishizawa/GENIAC/defactoring/Tanuki_pretraining/training/checkpoints/8b-iter0126000_tp${TENSOR_PARALLEL_SIZE}_pp${PIPELINE_PARALLEL_SIZE}

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/home/knishizawa/GENIAC/defactoring/Tanuki_pretraining/training/HF_data/Tanuki-8B-base-v1.0/tokenizer.json

cd /home/knishizawa/GENIAC/defactoring/Tanuki_pretraining/Megatron-LM
# convert
python /home/knishizawa/GENIAC/defactoring/Tanuki_pretraining/Megatron-LM/tools/checkpoint/convert.py \
  --model-type GPT \
  --loader llama3_hf \
  --saver mcore \
  --target-tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --target-pipeline-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --load-dir ${HF_CHECKPOINT_DIR} \
  --save-dir ${MEGATRON_CHECKPOINT_DIR} \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --bf16 \
  --saver-transformer-impl "transformer_engine"
