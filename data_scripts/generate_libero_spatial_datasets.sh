#!/bin/bash
# Usage: bash data_scripts/generate_libero_spatial_datasets.sh <suite>
# <suite>: libero_spatial | libero_object | libero_goal
export MUJOCO_GL=egl;

SUITE=${1:-libero_spatial}
LIBERO_DATASETS_ROOT=/home/ralf_roemer/projects/gym-libero/gym_libero/libero/datasets

for i in $(seq 0 9); do
    python ./data_scripts/regenerate_libero_hdf5_dataset_as_hf.py \
        --libero_task_suite $SUITE \
        --libero_raw_data_dir $LIBERO_DATASETS_ROOT/$SUITE \
        --task_ids $i \
        --repo_id continuallearning/${SUITE}_image_task_$i \
        --push_to_hub True
done
