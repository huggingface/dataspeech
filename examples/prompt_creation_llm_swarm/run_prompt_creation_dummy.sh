#!/usr/bin/env bash

python run_prompt_creation_llm_swarm.py \
  --dataset_name "stable-speech/libritts-r-tags-and-text" \
  --dataset_config_name "clean" \
  --dataset_split_name "train.clean.100" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --output_dir "./libritts-r-descriptions"
