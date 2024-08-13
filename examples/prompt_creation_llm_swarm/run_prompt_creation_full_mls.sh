#!/usr/bin/env bash

python ./run_prompt_creation_llm_swarm.py \
  --dataset_name 'ylacombe/mls-eng-text-tags-v5' \
  --dataset_config_name 'default' \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --num_instances "8" \
  --output_dir "./tmp_mls_with_accent" \
  --push_to_hub \
  --hub_dataset_id 'ylacombe/mls-eng-descriptions-v5' \
  --temperature 1.2 \
  --is_new_speaker_prompt \
  --speaker_id_column 'speaker_id' \
  --speaker_ids_to_name_json ./examples/prompt_creation/speaker_ids_to_names.json \
  --accent_column 'accent'