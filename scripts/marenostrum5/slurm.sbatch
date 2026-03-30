#!/bin/bash
#SBATCH --job-name=MN5_pi05_bowl
#SBATCH --account=ehpc660
#SBATCH --qos=acc_ehpc
#SBATCH --chdir=.
#SBATCH --output=lerobot_%j.out
#SBATCH --error=lerobot_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00

# Required for srun to respect cpus-per-task
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# Activate environment
source /gpfs/projects/ehpc660/oliver_hausdoerfer/envs/lerobot_realWorldDRL/bin/activate

# Offline mode
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export WANDB_API_KEY=$(awk '/api.wandb.ai/{getline; getline; print $2}' ~/.netrc)

# Cache directories
export HF_HUB_CACHE=/gpfs/projects/ehpc660/oliver_hausdoerfer/runs_root/cache/huggingface/hub
export HF_DATASETS_CACHE=/gpfs/projects/ehpc660/oliver_hausdoerfer/runs_root/cache/huggingface/datasets
export HF_LEROBOT_HOME=/gpfs/projects/ehpc660/oliver_hausdoerfer/runs_root/cache/huggingface/lerobot

# Paths
export PYTHONPATH=/home/tum/tum485846/lerobot_realWorldDRL/lerobot_realWorldDRL/lerobot/src:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/apps/ACC/GCC/15.2.0_nvptx-tools/lib64:/gpfs/projects/ehpc660/oliver_hausdoerfer/envs/lerobot_realWorldDRL/lib:${LD_LIBRARY_PATH:-}

# Modules
module load ffmpeg/7.0.1-gcc

# Run training
bash scripts/marenostrum5/train_pi05_gpu4.sh
