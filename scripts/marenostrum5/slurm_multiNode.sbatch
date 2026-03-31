#!/bin/bash
#SBATCH --job-name=pi05_libero
#SBATCH --account=ehpc660
#SBATCH --qos=acc_ehpc
#SBATCH --chdir=.
#SBATCH --output=lerobot_%j.out
#SBATCH --error=lerobot_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=11:00:00


# Required for srun to respect cpus-per-task
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

source scripts/marenostrum5/setup_run.sh

# Multi-node
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# NCCL tuning for inter-node H100 communication
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export OMP_NUM_THREADS=8


srun bash scripts/marenostrum5/train_pi05_gpu8_libero.sh

