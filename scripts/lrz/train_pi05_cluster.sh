# pi05, xvla: when training the entire model batch size 64 is the maximum with 80GB VRAM

lerobot-train \
    --dataset.repo_id=real_0_put_bowl_pi05 \
	--dataset.root=datasets/continuallearning/real_0_put_bowl_pi05 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_training_multigpu_$(date +%Y%m%d_%H%M%S) \
    --job_name=bowl \
    --policy.repo_id=your_repo_id \
	--policy.push_to_hub=false \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.optimizer_lr=1e-5 \
    --steps=50_000 \
    --save_freq=5_000 \
    --policy.device=cuda \
    --policy.use_amp=true \
    --batch_size=32 \
	--rename_map='{"observation.state_pi05": "observation.state"}'