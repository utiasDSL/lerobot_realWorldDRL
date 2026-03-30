# This file is mostly according to https://docs.google.com/document/d/1tSZCqn0r40dMepAQ01H-_1OU0XOaj2prxggCDaqnOUs/edit?usp=sharing

############### SETUP ###############
# @local pack the conda env (ignoring editable packages)
conda-pack -p /home/admin_07/miniconda3/envs/lerobot_realWorldDRL \
  -o scripts/marenostrum5/lerobot_realWorldDRL_env.tar.gz \
  --ignore-editable-packages

# @local transfer to cluster (storage partition)
scp \
	scripts/marenostrum5/lerobot_realWorldDRL_env.tar.gz \
	tum485846@transfer1.bsc.es:/gpfs/projects/ehpc660/oliver_hausdoerfer/envs

# @CLUSTER: unpack the environment and cleanup
# unpack: tar -xzf lerobot_realWorldDRL_env.tar.gz -C ./lerobot_realWorldDRL
# activate: source /gpfs/projects/ehpc660/oliver_hausdoerfer/envs/lerobot_realWorldDRL/bin/activate
# cleanup: conda-unpack

# @local copy the source code to the cluster
# ATTENTION: the --delete flag will delete all files on the cluster not present locally!!!
rsync -azP \
  --exclude '.claude' \
  --exclude '/lerobot_realWorldDRL/lerobot/datasets/' \
  --exclude '/lerobot_realWorldDRL/lerobot/outputs/' \
  /home/admin_07/project_repos/lerobot_realWorldDRL \
  tum485846@transfer1.bsc.es:/home/tum/tum485846/lerobot_realWorldDRL

# @CLUSTER: install source code into the environment that has previously been activated and unpacked (see above)
# pip install --no-deps --no-build-isolation -e ./lerobot_realWorldDRL/lerobot_realWorldDRL/lerobot/


# Additional things you might need for your runs:
# @local Copy dataset. This requires that the path exists on the cluster first.
scp -r \
	/home/admin_07/project_repos/lerobot_realWorldDRL/lerobot/datasets/continuallearning/real_0_put_bowl_pi05 \
	tum485846@transfer1.bsc.es:/gpfs/projects/ehpc660/oliver_hausdoerfer/runs_root/cache/huggingface/datasets/continuallearning/real_0_put_bowl_pi05

# Copy policy checkpoints from hf.
# You can download it like this locally first if you need to:
# pip install huggingface_hub
# huggingface-cli download lerobot/pi05_base
scp -r \
	/home/admin_07/.cache/huggingface/hub/models--lerobot--pi05_base \
	tum485846@transfer1.bsc.es:/gpfs/projects/ehpc660/oliver_hausdoerfer/runs_root/cache/huggingface/hub
scp -r ~/.cache/huggingface/hub/models--google--paligemma-3b-pt-224 \
  tum485846@transfer1.bsc.es:/gpfs/projects/ehpc660/oliver_hausdoerfer/runs_root/cache/huggingface/hub


############### START RUN ###############
# @CLUSTER
# NOTE for all runs set lerobot --wandb.mode=offline
# NOTE for all runs set lerobot --dataset.video_backend=pyav

# Save your wandb API key in ~/.netrc like this:
# cat >> ~/.netrc << 'EOF'
# machine api.wandb.ai
#   login user
#   password <YOUR_KEY_HERE>
# EOF
# chmod 600 ~/.netrc

##### Option 1: interactive run (debugging, testing)
# NOTE: the following instructions are very project specific and might vary from the instructions in the GDoc linked above. Please adapt accordingly for your project and cluster.

# @LoginNode Request resource
# salloc -A ehpc660 -t 00:30:00 -q acc_debug -n 1 -c 40 --gres=gpu:2
# -> wait for allocation
# srun --pty bash
# -> both /gpfs and ~/ are already available inside the job


# @ComputeNode start run
source /gpfs/projects/ehpc660/oliver_hausdoerfer/envs/lerobot_realWorldDRL/bin/activate
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline 
export WANDB_API_KEY=$(awk '/api.wandb.ai/{getline; getline; print $2}' ~/.netrc)

export HF_HUB_CACHE=/gpfs/projects/ehpc660/oliver_hausdoerfer/runs_root/cache/huggingface/hub
export HF_DATASETS_CACHE=/gpfs/projects/ehpc660/oliver_hausdoerfer/runs_root/cache/huggingface/datasets
export HF_LEROBOT_HOME=/gpfs/projects/ehpc660/oliver_hausdoerfer/runs_root/cache/huggingface/lerobot

export PYTHONPATH=/home/tum/tum485846/lerobot_realWorldDRL/lerobot_realWorldDRL/lerobot/src:${PYTHONPATH:-}
export LD_LIBRARY_PATH=/apps/ACC/GCC/15.2.0_nvptx-tools/lib64:/gpfs/projects/ehpc660/oliver_hausdoerfer/envs/lerobot_realWorldDRL/lib:${LD_LIBRARY_PATH:-}

module load ffmpeg/7.0.1-gcc

bash train.sh

# @ComputeNode monitor GPU usage (aim for 100% util. and mem.)
# ssh tum485846@alogin1.bsc.es
# ssh as02r3b19    # whatever node squeue shows
# watch -n 2 nvidia-smi


##### Option 2: run production job
# @LoginNode submit and sbatch job. See sbatch scipt files in this project.


############### AFTER RUN ###############

#@local Copy runs to local and upload to wandb
rsync -azP \
  tum485846@transfer1.bsc.es:/gpfs/projects/ehpc660/oliver_hausdoerfer/outputs/ \
  /home/admin_07/project_repos/lerobot_realWorldDRL/lerobot/outputs/mn5_outputs
cd /home/admin_07/project_repos/lerobot_realWorldDRL/lerobot/outputs/mn5_outputs
# python
for run in $(find . -type d -name "offline-run-*"); do
    echo "Syncing $run"
    wandb sync --exclude-globs "*.safetensors,*.pt,*.bin,*.pth" "$run"
done
