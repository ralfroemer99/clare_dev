#!/bin/bash
#SBATCH -p mcml-hgx-h100-94x4
#SBATCH -q mcml
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH -o /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/slurm_logs/clare_dev/scratch_franka.out
#SBATCH -e /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/slurm_logs/clare_dev/scratch_franka.err

source ~/.bashrc

srun --ntasks-per-node=1 \
    --container-image="${ENROOT}/clare_real.sqsh" \
    --container-mounts="${DSS}/ralf_roemer/clare_dev:/workspace/clare_dev,${DSS}/ralf_roemer/lerobot_yi:/workspace/lerobot_yi" \
    bash -c "export WANDB_API_KEY=${WANDB_API_KEY} && \
             export HF_TOKEN=${HF_TOKEN} && \
             export HF_HOME=${DSS}/ralf_roemer/.cache/huggingface && \
             cd /workspace/clare_dev && \
             bash ./bash/real/scratch/scratch_franka.sh"
