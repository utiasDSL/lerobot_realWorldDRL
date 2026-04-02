#libero_spatial,libero_object,libero_goal,

# fullfinetuned model, run1
# lerobot-eval \
#   --env.type=libero \
#   --env.task=libero_10 \
#   --eval.batch_size=1 \
#   --eval.n_episodes=10 \
#   --policy.path=outputs/mn5_outputs/pi05_libero_20260331_175300/checkpoints/last/pretrained_model \
#   --policy.n_action_steps=10 \
#   --policy.compile_model=false \
#   --output_dir=./eval_logs/pi05_libero_20260331_175300/ \
#   --env.max_parallel_tasks=1

# lerobot model finetuned on libero
lerobot-eval \
  --env.type=libero \
  --env.task=libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.path=lerobot/pi05_libero_finetuned \
  --policy.n_action_steps=10 \
  --output_dir=./eval_logs/pi05_libero_finetuned/ \
  --policy.compile_model=false \
  --env.max_parallel_tasks=1 \
  --policy.dtype=bfloat16


