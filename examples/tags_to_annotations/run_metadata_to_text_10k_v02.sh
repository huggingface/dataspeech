#!/usr/bin/env bash

python ./scripts/metadata_to_text.py "ylacombe/mls-eng-10k-tags+ylacombe/libritts_r_tags+ylacombe/libritts_r_tags" \
    --configuration "default+clean+other" \
    --output_dir "./tmp_mls+./tmp_tts_clean+./tmp_tts_other" \
    --cpu_num_workers "8" \
    --leading_split_for_bins "train" \
    --plot_directory "./plots/" \
    --save_bin_edges "./examples/tags_to_annotations/v02_bin_edges.json" \
    --path_to_text_bins ".examples/tags_to_annotations/v02_text_bins.json" \
    --pitch_std_tolerance "1.5"\
    --reverberation_std_tolerance "8."\
    --speech_monotony_std_tolerance "2."\
    --speaking_rate_std_tolerance "5.5"\
    --snr_std_tolerance "3.5"\
    --only_save_plot