#!/usr/bin/env bash

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8 run_prompt_creation.py \
  --dataset_name "parler-tts/libritts-r-tags-and-text" \
  --dataset_config_name "clean" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 4 \
  --output_dir "./" \
  --push_to_hub \
  --is_new_speaker_prompt \
  --speaker_id_column 'speaker_id' \
  --hub_dataset_id "parler-tts/libritts-r-tags-and-text-generated" \
  --speaker_ids_to_name_json ./examples/prompt_creation/speaker_ids_to_names.json \

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8 run_prompt_creation.py \
  --dataset_name "parler-tts/libritts-r-tags-and-text" \
  --dataset_config_name "other" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 4 \
  --output_dir "./" \
  --push_to_hub \
  --is_new_speaker_prompt \
  --speaker_id_column 'speaker_id' \
  --hub_dataset_id "parler-tts/libritts-r-tags-and-text-generated" \
  --speaker_ids_to_name_json ./examples/prompt_creation/speaker_ids_to_names.json \

