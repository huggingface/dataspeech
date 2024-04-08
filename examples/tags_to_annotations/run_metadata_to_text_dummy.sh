#!/usr/bin/env bash

python ./scripts/metadata_to_text.py "ylacombe/mls-eng-10k-tags+ylacombe/libritts_r_tags+ylacombe/libritts_r_tags" \
    --configuration "default+clean+other" \
    --dump_folder_path "./tmp_mls+./tmp_tts_clean+./tmp_tts_other" \
    --cpu_num_workers "8" \
    --leading_split_for_bins "train" \
    --plot_directory "./plots/"
