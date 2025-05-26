#!/bin/bash

NEMO_FILE=$1

cp -p /scripts/megatron_synth_topic_multiturn.py /opt/NeMo/examples/nlp/language_modeling/megatron_synth_topic_multiturn.py
cp -p /scripts/megatron_synth_topic_multiturn.yaml /opt/NeMo/examples/nlp/language_modeling/conf/megatron_synth_topic_multiturn.yaml

HYDRA_FULL_ERROR=1 /usr/bin/python3 /opt/NeMo/examples/nlp/language_modeling/megatron_synth_topic_multiturn.py \
        gpt_model_file="$NEMO_FILE" \
        pipeline_model_parallel_split_rank=0 \
        tensor_model_parallel_size=8 \
        trainer.precision=bf16 \
        pipeline_model_parallel_size=2 \
        trainer.devices=8 \
        trainer.num_nodes=2 \
        task_jsonl="/data/math_task.jsonl" \
        topic_jsonl="/data/math_topic.jsonl" \
        outputs_jsonl="/data/outputs.topic.math.1.jsonl" \
        batch_size=128 \
        num_samples=100 \
        seed=1 \
        inference.greedy=False \
        inference.temperature=0.7 \
        inference.top_k=0 \
        inference.top_p=1.0 \
        inference.repetition_penalty=1.0 \
        inference.tokens_to_generate=512 \
        inference.min_tokens_to_generate=0 \
        inference.add_BOS=True \
        inference.all_probs=False \
        inference.compute_logprob=False
