#!/bin/bash

NEMO_FILE=$1

# cp -p /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_eval.py /scripts/megatron_gpt_eval.org.py
# cp -p /opt/NeMo/examples/nlp/language_modeling/conf/megatron_gpt_inference.yaml /scripts/megatron_gpt_inference.org.yaml

cp -p /scripts/megatron_gpt_eval.py /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_eval.py
cp -p /scripts/megatron_gpt_inference.yaml /opt/NeMo/examples/nlp/language_modeling/conf/megatron_gpt_inference.yaml

HYDRA_FULL_ERROR=1 /usr/bin/python3 /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_eval.py \
        gpt_model_file="$NEMO_FILE" \
        pipeline_model_parallel_split_rank=0 \
        tensor_model_parallel_size=8 \
        trainer.precision=bf16 \
        pipeline_model_parallel_size=2 \
        trainer.devices=8 \
        trainer.num_nodes=2 \
        prompts_jsonl="/data/prompts.jsonl" \
        outputs_jsonl="/data/outputs.jsonl" \
        batch_size=128 \
        chunk_size=128 \
        seed=-1 \
        inference.greedy=False \
        inference.temperature=1.0 \
        inference.top_k=0 \
        inference.top_p=1.0 \
        inference.repetition_penalty=1.0 \
        inference.tokens_to_generate=1024 \
        inference.min_tokens_to_generate=0 \
        inference.add_BOS=True \
        inference.all_probs=False \
        inference.compute_logprob=False
