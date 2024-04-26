#!/usr/bin/env bash

python ./scripts/metadata_to_text.py \
    "ylacombe/jenny-tts-tags" \
    --repo_id "jenny-tts-tags" \
    --configuration "default" \
    --cpu_num_workers "8" \
    --path_to_bin_edges "./examples/tags_to_annotations/v01_bin_edges.json" \
    --avoid_pitch_computation
