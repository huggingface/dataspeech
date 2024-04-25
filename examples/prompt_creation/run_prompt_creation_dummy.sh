#!/usr/bin/env bash

python run_prompt_creation.py \
  --dataset_name "ylacombe/libritts_r_tags_and_text" \
  --dataset_config_name "clean" \
  --dataset_split_name "dev.clean" \
  --model_name_or_path "hf-internal-testing/tiny-random-LlamaForCausalLM" \
  --per_device_eval_batch_size 2 \
  --attn_implementation "sdpa" \
  --torch_compile \
  --max_eval_samples 8 \
  --dataloader_num_workers 0 \
  --output_dir "./"
