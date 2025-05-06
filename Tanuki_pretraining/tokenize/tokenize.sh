#!/bin/bash

# Conda 環境をアクティベート
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate textprocess

export PYTHONPATH=/home/ubuntu/Tanuki_pretraining/Megatron-LM:$PYTHONPATH

# tokenize
output_prefix=$(yq -r '.output_prefix' ./tokenize/tokenize_config.yaml)
input_jsonl=$(yq -r '.input' ./tokenize/tokenize_config.yaml)
seq_length=$(yq -r '.seq_length' ./tokenize/tokenize_config.yaml)
max_workers=$(yq -r '.max_workers' ./tokenize/tokenize_config.yaml)
input_tokenizer_file=$(yq -r '.input_tokenizer_file' ./tokenize/tokenize_config.yaml)

echo "tokenizer-model: ${input_tokenizer_file}"

python ./tokenize/tokenize_preprocess.py \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-model ${input_tokenizer_file} \
    --input  ${input_jsonl} \
    --output-prefix ${output_prefix} \
    --dataset-impl mmap \
    --seq-length ${seq_length} \
    --max_workers ${max_workers} \
    --append-eod
echo ""

# Conda 環境をデアクティベート
# conda deactivate