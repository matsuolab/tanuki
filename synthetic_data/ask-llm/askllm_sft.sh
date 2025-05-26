#!/bin/bash

# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodelist=slurm0-a3-ghpc-0
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160GB
#SBATCH --job-name=askllm
#SBATCH --output=logs/%x_%j.log


source $EXP_HOME/miniconda3/etc/profile.d/conda.sh
conda activate askllm311


input_jsonl="data/synth-topic-jp-reasoning-nemotron-4.20240807+20240808.fasttext.jsonl"
output_jsonl="data/synth-topic-jp-reasoning-nemotron-4.20240807+20240808.askllm_q1.jsonl"
output_score_column="q1_solvable_score"
max_tokens=2048
dataset_path=""
dataset_name="default"
dataset_split="train"
target_columns="messages"
messages_indices="0"
separator="\n\n"
sort="True"
reverse="True"
# model_id="cyberagent/calm3-22b-chat"
# model_id="/storage7/tmp/ota/calm3-22b-chat"
# model_id="Rakuten/RakutenAI-7B-instruct"
# model_id="microsoft/Phi-3-medium-128k-instruct"
model_id="microsoft/Phi-3-mini-4k-instruct"
cache_dir="/storage5/shared/ota_s/askllm/hf_cache"
log_interval=1000
wandb_project="askllm"
wandb_entity="weblab-geniac1"
wandb_name="synth-topic-jp-reasoning-nemotron-4"


time python scripts/askllm_sft.py \
    --input_jsonl="$input_jsonl" \
    --output_jsonl="$output_jsonl" \
    --output_score_column="$output_score_column" \
    --max_tokens="$max_tokens" \
    --dataset_path="$dataset_path" \
    --dataset_name="$dataset_name" \
    --dataset_split="$dataset_split" \
    --target_columns="$target_columns" \
    --messages_indices="$messages_indices" \
    --separator="$separator" \
    --sort="$sort" \
    --reverse="$reverse" \
    --model_id="$model_id" \
    --cache_dir="$cache_dir" \
    --log_interval="$log_interval" \
    --wandb_project="$wandb_project" \
    --wandb_entity="$wandb_entity" \
    --wandb_name="$wandb_name" \
