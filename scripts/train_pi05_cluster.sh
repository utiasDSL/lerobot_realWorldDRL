# pi05, xvla: when training the entire model batch size 64 is the maximum with 80GB VRAM

cluster
    accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    $(which lerobot-train) \
    --dataset.repo_id=real_0_put_bowl_pi05 \
	--dataset.root=datasets/continuallearning/real_0_put_bowl_pi05 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05_training_$(date +%Y%m%d_%H%M%S) \
    --job_name=bowl \
    --policy.repo_id=your_repo_id \
	--policy.push_to_hub=false \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=false \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --steps=10_000 \
    --policy.device=cuda \
    --batch_size=64 \
	--rename_map='{"observation.state_pi05": "observation.state"}'