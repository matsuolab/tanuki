#!/bin/bash

# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodelist=slurm0-a3-ghpc-0
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --job-name=split_by_topic
#SBATCH --output=logs/%x_%j.log


source $EXP_HOME/miniconda3/etc/profile.d/conda.sh
conda activate askllm311


input_jsonl="data/synth-topic-jp-basic-math-reasoning-calm3.20240809.fasttext.jsonl"
output1_jsonl="data/synth-topic-jp-basic-math-calm3.20240809.fasttext.jsonl"
output2_jsonl="data/synth-topic-jp-basic-reasoning-calm3.20240809.fasttext.jsonl"
dataset_path=""
dataset_name="default"
dataset_split="train"
# cache_dir="/storage5/shared/ota_s/askllm/hf_cache"
cache_dir=""


time python scripts/split_by_topic.py \
    --input_jsonl="$input_jsonl" \
    --output1_jsonl="$output1_jsonl" \
    --output2_jsonl="$output2_jsonl" \
    --dataset_path="$dataset_path" \
    --dataset_name="$dataset_name" \
    --dataset_split="$dataset_split" \
    --cache_dir="$cache_dir" \
