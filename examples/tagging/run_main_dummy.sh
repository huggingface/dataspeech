#!/usr/bin/env bash

python main.py "blabble-io/libritts_r" \
    --configuration "dev" \
    --dump_folder_path ./tmp_libritts_r_dev/ \
    --text_column_name "text_normalized" \
    --audio_column_name "audio" \
    --cpu_num_workers 8 \
    --num_workers_per_gpu 4 \
    --rename_column \