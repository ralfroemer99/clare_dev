#!/bin/bash
# Create ER replay datasets for Libero-40 (Libero-10 -> Goal -> Spatial -> Object).
# Each replay dataset contains 5 episodes sampled from each task seen so far.

set -e

NUM_EP=5
SCRIPT=./lerobot_lsy/src/lerobot/scripts/util/create_er_dataset.py
PREFIX=continuallearning/libero_40_image_task

SOURCE_REPOS=(
    continuallearning/libero_10_image_task_0
    continuallearning/libero_10_image_task_1
    continuallearning/libero_10_image_task_2
    continuallearning/libero_10_image_task_3
    continuallearning/libero_10_image_task_4
    continuallearning/libero_10_image_task_5
    continuallearning/libero_10_image_task_6
    continuallearning/libero_10_image_task_7
    continuallearning/libero_10_image_task_8
    continuallearning/libero_10_image_task_9
    continuallearning/libero_goal_image_task_0
    continuallearning/libero_goal_image_task_1
    continuallearning/libero_goal_image_task_2
    continuallearning/libero_goal_image_task_3
    continuallearning/libero_goal_image_task_4
    continuallearning/libero_goal_image_task_5
    continuallearning/libero_goal_image_task_6
    continuallearning/libero_goal_image_task_7
    continuallearning/libero_goal_image_task_8
    continuallearning/libero_goal_image_task_9
    continuallearning/libero_spatial_image_task_0
    continuallearning/libero_spatial_image_task_1
    continuallearning/libero_spatial_image_task_2
    continuallearning/libero_spatial_image_task_3
    continuallearning/libero_spatial_image_task_4
    continuallearning/libero_spatial_image_task_5
    continuallearning/libero_spatial_image_task_6
    continuallearning/libero_spatial_image_task_7
    continuallearning/libero_spatial_image_task_8
    continuallearning/libero_spatial_image_task_9
    continuallearning/libero_object_image_task_0
    continuallearning/libero_object_image_task_1
    continuallearning/libero_object_image_task_2
    continuallearning/libero_object_image_task_3
    continuallearning/libero_object_image_task_4
    continuallearning/libero_object_image_task_5
    continuallearning/libero_object_image_task_6
    continuallearning/libero_object_image_task_7
    continuallearning/libero_object_image_task_8
    continuallearning/libero_object_image_task_9
)

# Before task 1: replay of task 0 only
python $SCRIPT \
    --repo_ids=${SOURCE_REPOS[0]} \
    --num_episodes=$NUM_EP \
    --merged_repo_id=${PREFIX}_0_er \
    --push_to_hub

# Before task N+1 (N=1..38): replay of tasks 0..N
for N in $(seq 1 38); do
    REPO_IDS=$(IFS=,; echo "${SOURCE_REPOS[*]:0:$((N+1))}")
    python $SCRIPT \
        --repo_ids=$REPO_IDS \
        --num_episodes=$NUM_EP \
        --merged_repo_id=${PREFIX}_0_${N} \
        --push_to_hub
done
