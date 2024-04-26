#!/usr/bin/env bash

python run_prompt_creation_llm_swarm.py \
  --dataset_name "stable-speech/libritts-r-tags-and-text" \
  --dataset_config_name "clean" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.1" \
  --output_dir "./" \
  --max_eval_samples 4100 \
  --debug_endpoint "http://26.0.173.202:53138"
