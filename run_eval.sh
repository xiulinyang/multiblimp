#!/bin/bash

# Define basic settings
model_name="xiulinyang/GPT-EN-5k"
data_dir="hf_cache/eng/"
src_dir="multiblimp"
results_dir="multiblimp_results/"
cache_dir="hf_cache/"

# Loop over checkpoints
for checkpoint in $(seq 0 100 1200); do
    echo "Evaluating checkpoint: $checkpoint"

    python scripts/lm_eval/eval_model.py \
        --model "$model_name" \
        --revision "checkpoint-$checkpoint" \
        --data_dir "$data_dir" \
        --src_dir "$src_dir" \
        --results_dir "$results_dir" \
        --cache_dir "$cache_dir"
done
