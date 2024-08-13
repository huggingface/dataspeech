#!/usr/bin/env bash
python ./scripts/run_prompt_creation.py \
  --speaker_name "Jenny" \
  --is_single_speaker \
  --is_new_speaker_prompt \
  --dataset_name "ylacombe/jenny-tts-tags-v1" \
  --dataset_config_name "default" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 128 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 8 \
  --output_dir "./tmp_jenny" \
  --load_in_4bit \
  --push_to_hub \
  --hub_dataset_id "jenny-tts-tagged-v1" \
  --preprocessing_num_workers 48 \
  --dataloader_num_workers 24