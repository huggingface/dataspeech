#!/usr/bin/env bash

python ./scripts/metadata_to_text.py \
    "ylacombe/jenny-tts-tags" \
    --repo_id "ylacombe/jenny-tts-tags" \
    --output_dir "./tmp_jenny" \
    --configuration "default" \
    --cpu_num_workers "8" \
    --leading_split_for_bins "train" \
    --path_to_bin_edges "./examples/tags_to_annotations/v01_bin_edges.json" \
    --avoid_pitch_computation
