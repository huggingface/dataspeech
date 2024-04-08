#!/usr/bin/env bash

python main.py "ylacombe/english_dialects" \
    --configuration "midlands_male" \
    --dump_folder_path ./tmp_english_midlands/ \
    --text_column_name "text" \
    --audio_column_name "audio" \
    --cpu_num_workers 8 \
    --num_workers_per_gpu 4 \