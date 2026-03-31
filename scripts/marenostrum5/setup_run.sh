source /gpfs/projects/ehpc660/oliver_hausdoerfer/envs/lerobot_realWorldDRL/bin/activate
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline 
export WANDB_API_KEY=$(awk '/api.wandb.ai/{getline; getline; print $2}' ~/.netrc)

export HF_HUB_CACHE=/gpfs/projects/ehpc660/oliver_hausdoerfer/runs_root/cache/huggingface/hub
export HF_DATASETS_CACHE=/gpfs/projects/ehpc660/oliver_hausdoerfer/runs_root/cache/huggingface/datasets
export HF_LEROBOT_HOME=/gpfs/projects/ehpc660/oliver_hausdoerfer/runs_root/cache/huggingface/lerobot

export WANDB_DATA_DIR=/gpfs/scratch/ehpc637/oliver_hausdorfer/wandb_cache
export WANDB_ARTIFACTS_DIR=/gpfs/scratch/ehpc637/oliver_hausdorfer/wandb_cache

export PYTHONPATH=/home/tum/tum485846/lerobot_realWorldDRL/lerobot_realWorldDRL/lerobot/src:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/apps/ACC/GCC/15.2.0_nvptx-tools/lib64:/gpfs/projects/ehpc660/oliver_hausdoerfer/envs/lerobot_realWorldDRL/lib:${LD_LIBRARY_PATH:-}

# Cache for torch.compile. USE ONLY IF YOU KNOW WHAT YOU ARE DOING
export TORCHINDUCTOR_CACHE_DIR=/gpfs/scratch/ehpc637/oliver_hausdorfer/torch_inductor_cache
export TRITON_CACHE_DIR=/gpfs/scratch/ehpc637/oliver_hausdorfer/triton_cache
export TORCHINDUCTOR_FX_GRAPH_CACHE=1 

module load ffmpeg/7.0.1-gcc