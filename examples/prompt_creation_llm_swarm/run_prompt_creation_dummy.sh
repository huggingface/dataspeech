#!/usr/bin/env bash

python run_prompt_creation_llm_swarm.py \
  --dataset_name "ylacombe/libritts_r_tags_and_text" \
  --dataset_config_name "clean" \
  --dataset_split_name "dev.clean" \
  --model_name_or_path "meta-llama/Llama-2-7b-hf" \
  --output_dir "./" \
  --max_eval_samples 1999 \
  --debug_endpoint "http://26.0.173.246:59699"
