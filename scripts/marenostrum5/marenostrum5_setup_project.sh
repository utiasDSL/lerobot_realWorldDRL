# This file is mostly according to https://docs.google.com/document/d/1tSZCqn0r40dMepAQ01H-_1OU0XOaj2prxggCDaqnOUs/edit?usp=sharing

# pack the conda env (ignoring editable packages)
conda-pack -p /home/admin_07/miniconda3/envs/lerobot_realWorldDRL \
  -o scripts/marenostrum5/lerobot_realWorldDRL_env.tar.gz \
  --ignore-editable-packages

# transfer to cluster (storage partition)
scp \
	scripts/marenostrum5/lerobot_realWorldDRL_env.tar.gz \
	tum485846@transfer1.bsc.es:/gpfs/projects/ehpc660/oliver_hausdoerfer/envs

# @CLUSTER: unpack the environment and cleanup
# unpack: tar -xzf lerobot_realWorldDRL_env.tar.gz -C ./lerobot_realWorldDRL
# activate: source /gpfs/projects/ehpc660/oliver_hausdoerfer/envs/lerobot_realWorldDRL/bin/activate
# cleanup: conda-unpack

# copy the source code to the cluster
# ATTENTION: the --delete flag will delete all files on the cluster not present locally!!!
rsync -azP \
  --exclude '.*' \
  --exclude 'datasets/' \
  --exclude 'outputs/' \
  /home/admin_07/project_repos/lerobot_realWorldDRL \
  tum485846@transfer1.bsc.es:/home/tum/tum485846/lerobot_realWorldDRL

# @CLUSTER: install source code into the environment that has previously been activated and unpacked (see above)
# pip install --no-deps --no-build-isolation -e ./lerobot_realWorldDRL/lerobot_realWorldDRL/lerobot/


# Additional things you might need for your runs:
# Copy dataset. This requires that the path exists on the cluster first.
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