#!/bin/bash
# Create ER replay datasets for Libero Spatial and Libero Goal.
# Each dataset contains 5 episodes sampled from each task seen so far.
# Usage: bash bash/create_er_datasets.sh

set -e

NUM_EP=50
SCRIPT=./lerobot_lsy/src/lerobot/scripts/util/create_er_dataset.py

for SUITE in libero_spatial libero_goal; do
    PREFIX=continuallearning/${SUITE}_image_task

    # Task 1 replay: 5 episodes from task 0 only
    python $SCRIPT \
        --repo_ids=${PREFIX}_0 \
        --num_episodes=$NUM_EP \
        --merged_repo_id=${PREFIX}_0_er \
        --push_to_hub

    # Tasks 2–9 replay: 5 episodes from each task seen so far
    for N in 1 2 3 4 5 6 7 8; do
        REPO_IDS=$(seq 0 $N | xargs -I{} echo "${PREFIX}_{}" | paste -sd,)
        python $SCRIPT \
            --repo_ids=$REPO_IDS \
            --num_episodes=$NUM_EP \
            --merged_repo_id=${PREFIX}_0_${N} \
            --push_to_hub
    done
done
