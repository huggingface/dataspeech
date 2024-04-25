#!/usr/bin/env bash

python run_prompt_creation_llm_swarm.py \
  --dataset_name "ylacombe/libritts_r_tags_and_text" \
  --dataset_config_name "clean" \
  --dataset_split_name "dev.clean" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.1" \
  --output_dir "./" \
  --max_eval_samples 4 \
  --debug_endpoint "http://26.0.161.142:26460" \
  --per_instance_max_parallel_requests 2
