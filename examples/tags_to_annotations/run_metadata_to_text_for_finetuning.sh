#!/usr/bin/env bash

python ./scripts/metadata_to_text.py \
    "ylacombe/jenny-tts-tags-v1" \
    --repo_id "jenny-tts-tags-v1" \
    --configuration "default" \
    --cpu_num_workers "8" \
    --path_to_bin_edges "./examples/tags_to_annotations/v02_bin_edges.json" \
    --path_to_text_bins "./examples/tags_to_annotations/v02_text_bins.json" \
    --avoid_pitch_computation \
    --apply_squim_quality_estimation \

