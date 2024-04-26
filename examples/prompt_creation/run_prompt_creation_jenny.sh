#!/usr/bin/env bash
python ./scripts/run_prompt_creation_single_speaker.py \
  --speaker_name "Jenny" \
  --dataset_name "ylacombe/jenny-tts-tags" \
  --dataset_config_name "default" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 128 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 8 \
  --output_dir "./tmp_jenny" \
  --load_in_4bit \
  --push_to_hub \
  --hub_dataset_id "jenny-tts-10k-tagged" \
  --preprocessing_num_workers 48