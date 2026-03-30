# pi05, xvla: when training the entire model batch size 64 is the maximum with 80GB VRAM

# cluster
accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --mixed_precision="bf16"\
    $(which lerobot-train) \
    --dataset.repo_id=libero \
	--dataset.root=/gpfs/projects/ehpc660/oliver_hausdoerfer/runs_root/cache/huggingface/datasets/HuggingFaceVLA/libero \
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
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.optimizer_lr=1e-5 \
    --steps=30_000 \
    --save_freq=5_000 \
    --policy.device=cuda \
    --policy.use_amp=true \
    --batch_size=32 \
	--policy.n_action_steps=10 \
