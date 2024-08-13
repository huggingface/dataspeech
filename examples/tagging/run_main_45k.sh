#!/usr/bin/env bash

python main.py "blabble-io/libritts_r" \
    --configuration "clean" \
    --output_dir ./tmp_libritts_r_clean/ \
    --text_column_name "text_normalized" \
    --audio_column_name "audio" \
    --cpu_num_workers 32 \
    --num_workers_per_gpu 4 \
    --rename_column \
    --repo_id "ylacombe/libritts-r-text-tags-v3"\
    --apply_squim_quality_estimation \


python main.py "blabble-io/libritts_r" \
    --configuration "other" \
    --output_dir ./tmp_libritts_r_other/ \
    --text_column_name "text_normalized" \
    --audio_column_name "audio" \
    --cpu_num_workers 32 \
    --num_workers_per_gpu 4 \
    --rename_column \
    --repo_id "ylacombe/libritts-r-text-tags-v3"\
    --apply_squim_quality_estimation \

python main.py "parler-tts/mls_eng" \
    --output_dir ./tmp_mls_eng/ \
    --text_column_name "transcript" \
    --audio_column_name "audio" \
    --cpu_num_workers 32 \
    --num_workers_per_gpu 4 \
    --rename_column \
    --repo_id "ylacombe/mls-eng-tags-v4"\
    --apply_squim_quality_estimation \
