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
# cd /gpfs/projects/ehpc660/oliver_hausdoerfer/envs
# mkdir ./lerobot_realWorldDRL
# unpack: tar -xzf lerobot_realWorldDRL_env.tar.gz -C ./lerobot_realWorldDRL
# activate: source /gpfs/projects/ehpc660/oliver_hausdoerfer/envs/lerobot_realWorldDRL/bin/activate
# cleanup: conda-unpack

# @local copy the source code to the cluster
# ATTENTION: the --delete flag will delete all files on the cluster not present locally!!!
rsync -azP \
  # --delete \
  --exclude '.claude' \
  --exclude '/lerobot_realWorldDRL/lerobot/datasets/' \
  --exclude '/lerobot_realWorldDRL/lerobot/outputs/' \
  /home/admin_07/project_repos/lerobot_realWorldDRL \
  tum485846@transfer1.bsc.es:/home/tum/tum485846/lerobot_realWorldDRL

# @CLUSTER: install source code into the environment that has previously been activated and unpacked (see above)
# cd ~
# pip install --no-deps --no-build-isolation -e ./lerobot_realWorldDRL/lerobot_realWorldDRL/lerobot/


# Additional things you might need for your runs:
# @local Copy dataset. This requires that the path exists on the cluster first.
scp -r \
	/home/admin_07/project_repos/lerobot_realWorldDRL/lerobot/datasets/continuallearning/real_0_put_bowl \
	tum485846@transfer1.bsc.es:/gpfs/projects/ehpc660/oliver_hausdoerfer/runs_root/cache/huggingface/datasets/continuallearning/real_0_put_bowl

# Copy policy checkpoints from hf.
# You can download it locally first if you need to like this:
# pip install huggingface_hub
# huggingface-cli download lerobot/pi05_base
scp -r \
	/home/admin_07/.cache/huggingface/hub/models--lerobot--pi05_base \
	tum485846@transfer1.bsc.es:/gpfs/projects/ehpc660/oliver_hausdoerfer/runs_root/cache/huggingface/hub
scp -r ~/.cache/huggingface/hub/models--google--paligemma-3b-pt-224 \
  tum485846@transfer1.bsc.es:/gpfs/projects/ehpc660/oliver_hausdoerfer/runs_root/cache/huggingface/hub

# Libero assets and config: Currently already available on the cluster at /gpfs/projects/ehpc660/oliver_hausdoerfer/runs_root/cache/libero. Please copy the assets and config for your usage. Then, additionally set symbolic link because libero is fundamentally broken: 
# ln -s /gpfs/projects/ehpc660/oliver_hausdoerfer/runs_root/cache/libero/assets /gpfs/projects/ehpc660/oliver_hausdoerfer/envs/lerobot_realWorldDRL/lib/python3.12/site-packages/libero/libero/assets

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
source scripts/marenostrum5/setup_run.sh
bash train.sh

# @ComputeNode monitor GPU usage (aim for 100% util. and mem.)
# ssh tum485846@alogin1.bsc.es
# ssh as02r3b19    # whatever node squeue shows
# watch -n 2 nvidia-smi


##### Option 2: run production job
# @LoginNode submit and sbatch job. See sbatch scipt files in this project.


############### AFTER RUN ###############

#@local Copy runs to local and upload to wandb
REMOTE="tum485846@transfer1.bsc.es:/gpfs/projects/ehpc660/oliver_hausdoerfer/outputs"
LOCAL="/home/admin_07/project_repos/lerobot_realWorldDRL/lerobot/outputs/mn5_outputs"
RUNS=(
  "pi05_bowl_20260401_232730"
)
for run in "${RUNS[@]}"; do
  rsync -azP -L \
    --include="checkpoints/" \
    --include="checkpoints/last/" \
    --include="checkpoints/last/**" \
    --exclude="checkpoints/*" \
    "$REMOTE/$run/" \
    "$LOCAL/$run/"
done
cd /home/admin_07/project_repos/lerobot_realWorldDRL/lerobot/outputs/mn5_outputs
# python
for run in $(find . -type d -name "offline-run-*"); do
    echo "Syncing $run"
    wandb sync --exclude-globs "*.safetensors,*.pt,*.bin,*.pth" "$run"
done
