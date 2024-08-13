#!/usr/bin/env bash

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8 run_prompt_creation.py \
  --dataset_name "ylacombe/libritts-r-text-tags-v4" \
  --dataset_config_name "clean" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 4 \
  --output_dir "./libritts_r_descriptions_clean" \
  --push_to_hub \
  --is_new_speaker_prompt \
  --speaker_id_column 'speaker_id' \
  --speaker_ids_to_name_json ./examples/prompt_creation/speaker_ids_to_names.json \
  --hub_dataset_id "ylacombe/libritts-r-descriptions-10k-v5-without-accents"

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8 run_prompt_creation.py \
  --dataset_name "ylacombe/libritts-r-text-tags-v4" \
  --dataset_config_name "other" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 4 \
  --output_dir "./libritts_r_descriptions_other" \
  --push_to_hub \
  --is_new_speaker_prompt \
  --speaker_id_column 'speaker_id' \
  --speaker_ids_to_name_json ./examples/prompt_creation/speaker_ids_to_names.json \
  --hub_dataset_id "ylacombe/libritts-r-descriptions-10k-v5-without-accents"

accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8 run_prompt_creation.py \
  --dataset_name "ylacombe/mls-eng-text-tags-v5" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 4 \
  --output_dir "./mls-eng-descriptions" \
  --push_to_hub \
  --is_new_speaker_prompt \
  --speaker_id_column 'speaker_id' \
  --speaker_ids_to_name_json ./examples/prompt_creation/speaker_ids_to_names.json \
  --hub_dataset_id "parler-tts/mls-eng-speaker-descriptions"
