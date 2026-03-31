# pi05, xvla: when training the entire model batch size 64 is the maximum with 80GB VRAM

# 6 hours

# cluster
accelerate launch \
    --multi_gpu \
    --num_processes=8 \
    --num_machines=2 \
    --machine_rank=$SLURM_NODEID \
    --main_process_ip=$MASTER_ADDR \
    --main_process_port=$MASTER_PORT \
    --mixed_precision="bf16"\
    $(which lerobot-train) \
    --dataset.repo_id=real_0_put_bowl_pi05/ \
	--dataset.root=/gpfs/projects/ehpc660/oliver_hausdoerfer/runs_root/cache/huggingface/datasets/continuallearning/real_0_put_bowl_pi05/ \
    --dataset.video_backend=pyav \
    --policy.type=pi05 \
    --output_dir=/gpfs/projects/ehpc660/oliver_hausdoerfer/outputs/pi05_libero_$(date +%Y%m%d_%H%M%S) \
    --job_name=pi05_libero_$(date +%Y%m%d_%H%M%S) \
    --policy.repo_id=your_repo_id \
	--policy.push_to_hub=false \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --wandb.mode=offline \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.optimizer_lr=2.5e-5 \
    --steps=20_000 \
    --save_freq=4_000 \
    --policy.device=cuda \
    --policy.use_amp=true \
    --batch_size=80 \
	--policy.n_action_steps=10 \
    --policy.empty_cameras=1 \
    --rename_map='{"observation.state_pi05": "observation.state"}' \
