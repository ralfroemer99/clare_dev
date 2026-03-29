#!/bin/bash
#SBATCH -p mcml-hgx-h100-94x4
#SBATCH -q mcml
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH -o /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/slurm_logs/clare_dev/er_libero_40_%j.out
#SBATCH -e /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/slurm_logs/clare_dev/er_libero_40_%j.err

source ~/.bashrc

SEED=${1:-42}
PRETRAIN_PATH=${2:-/workspace/clare_dev/outputs/dit_flow_mt_libero_90_pretrain_new}

srun --ntasks-per-node=1 \
    --container-image="${ENROOT}/clare_sim.sqsh" \
    --container-mounts="${DSS}/ralf_roemer/clare_dev_main:/workspace/clare_dev,${DSS}/ralf_roemer/gym-libero:/workspace/gym-libero,${DSS}/ralf_roemer/.cache:/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/ralf_roemer/.cache" \
    --container-env="NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics" \
    bash -c "export WANDB_API_KEY=${WANDB_API_KEY} && \
             export HF_TOKEN=${HF_TOKEN} && \
             export HF_HOME=${DSS}/ralf_roemer/.cache/huggingface && \
             export PRETRAIN_PATH=${PRETRAIN_PATH} && \
             cd /workspace/clare_dev && \
             bash ./bash/er/er_libero_40.sh ${SEED}"
