#!/bin/bash

# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodelist=slurm0-a3-ghpc-0
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --job-name=lang_identifier
#SBATCH --output=logs/%x_%j.log


source $EXP_HOME/miniconda3/etc/profile.d/conda.sh
conda activate askllm311


# - Download the model from Hugging Face Hub
#
# pip install huggingface_hub[hf_transfer]
# HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
#   facebook/fasttext-language-identification model.bin  \
#   --repo-type=model --local-dir="." --local-dir-use-symlinks=False
# mkdir -p data
# mv model.bin data


input_jsonl="data/synth-topic-jp-basic-math-reasoning-calm3.20240809.filtered.jsonl"
output_jsonl="data/synth-topic-jp-basic-math-reasoning-calm3.20240809.fasttext.jsonl"
trash_jsonl="data/synth-topic-jp-basic-math-reasoning-calm3.20240809.fasttext_trash.jsonl"
output_score_column="fasttext_jp_score"
dataset_path=""
dataset_name="default"
dataset_split="train"
cache_dir="/storage5/shared/ota_s/askllm/hf_cache"
fasttext_path="data/model.bin"
fasttext_label="__label__jpn_Jpan"
target_columns="messages"
messages_indices="0,1,2,3"
separator="\n\n"
sort="True"
reverse="True"
threshold="0.07"


time python scripts/lang_identifier_sft.py \
    --input_jsonl="$input_jsonl" \
    --output_jsonl="$output_jsonl" \
    --trash_jsonl="$trash_jsonl" \
    --output_score_column="$output_score_column" \
    --dataset_path="$dataset_path" \
    --dataset_name="$dataset_name" \
    --dataset_split="$dataset_split" \
    --cache_dir="$cache_dir" \
    --fasttext_path="$fasttext_path" \
    --fasttext_label="$fasttext_label" \
    --target_columns="$target_columns" \
    --messages_indices="$messages_indices" \
    --separator="$separator" \
    --sort="$sort" \
    --reverse="$reverse" \
    --threshold="$threshold" \
