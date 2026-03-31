#!/bin/bash
#SBATCH --job-name=pi05_libero
#SBATCH --account=ehpc660
#SBATCH --qos=acc_ehpc
#SBATCH --chdir=.
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=11:00:00


# Required for srun to respect cpus-per-task
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

source scripts/marenostrum5/setup_run.sh

# Run training
bash scripts/marenostrum5/train_pi05_gpu4_libero.sh
