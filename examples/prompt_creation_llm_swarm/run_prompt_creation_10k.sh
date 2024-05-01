#!/usr/bin/env bash

python run_prompt_creation_llm_swarm.py \
  --dataset_name "ylacombe/mls-eng-10k-text-tags-v2" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --num_instances "2" \
  --output_dir "./mls-eng-10k-descriptions-v2" \
  --push_to_hub \
  --hub_dataset_id "stable-speech/mls-eng-10k-descriptions-v2"
