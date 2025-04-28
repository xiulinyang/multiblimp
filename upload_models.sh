#!/bin/bash

# Configurations
repo_id="your-username/your-model-name"
start=0
end=1200
step=100

# Log into huggingface (optional if not done)
# huggingface-cli login --token $hf_token

# Clone your repo
git clone https://huggingface.co/$repo_id
cd $(basename "$repo_id")

# Loop over checkpoints
for checkpoint in $(seq $start $step $end)
do
    branch_name="checkpoint-${checkpoint}"
    echo "Pushing branch: $branch_name"

    # Create and checkout new branch
    git checkout -b $branch_name

    # Copy or move your checkpoint files into this directory
    # (assumes your checkpoints are saved somewhere already)
    cp -r /path/to/your/checkpoints/$checkpoint/* ./

    # Stage, commit, push
    git add .
    git commit -m "Add checkpoint at step $checkpoint"
    git push origin $branch_name

    # Clean up before next iteration
    git reset --hard main
    git clean -fd
    git checkout main
done

echo "All checkpoints pushed!"

