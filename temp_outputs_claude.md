This machine state captured. Compare these with workstation A:

HF cache ($HF_HOME=/home/maxchr/data/hf):

snapshot commit: a017a25b4dc9163eefd5105cff5d8c84fe6b7dfd
model.safetensors blob: ec4019eb...a9d825
normalizer blob (shared pre/post): 5447f2c7...e6582f
config.json blob: 78a930ea...9609a
Local dataset meta (datasets/.../meta/):

stats.json sha256: 4074a030...931281
info.json sha256: 6308665a...68b6b6
lerobot editable (/home/maxchr/repos/lerobot_realWorldDRL):

HEAD 84a5477a, branch main
dirty: src/lerobot/policies/groot/groot_n1.py (groot only — not pi05 path, irrelevant to symptom)
Run on workstation A (same env $HF_HOME may differ — adjust path):


echo $HF_HOME $HF_HUB_CACHE
cat $HF_HUB_CACHE/models--OliverHausdoerfer--pi05_stack_lego_simple_20260418/refs/main
ls -la $HF_HUB_CACHE/models--OliverHausdoerfer--pi05_stack_lego_simple_20260418/snapshots/*/
sha256sum datasets/OliverHausdoerfer/stack_lego_simple_v1_filtered_new_state_fixed_stats/meta/stats.json \
          datasets/OliverHausdoerfer/stack_lego_simple_v1_filtered_new_state_fixed_stats/meta/info.json
cd /home/maxchr/repos/lerobot_realWorldDRL && git rev-parse HEAD && git status --short

========== TEMP DEBUG: raw sample ==========
  sample['observation.images.primary'] shape=(3, 256, 256) dtype=torch.float32 min=0 max=1 mean=0.574
  sample['observation.images.wrist'] shape=(3, 256, 256) dtype=torch.float32 min=0 max=0.8196 mean=0.421
  sample['observation.state'] shape=(7,) dtype=torch.float32 min=-0.1752 max=0.2267 mean=-0.005789
  sample['observation.state.gripper'] shape=() dtype=torch.float32 min=0.2267 max=0.2267 mean=0.2267
  sample['observation.state.sensors_bota_ft_sensor'] shape=(6,) dtype=torch.float32 min=-0.1752 max=0.05842 mean=-0.04453
  sample['action'] shape=(7,) dtype=torch.float32 min=-4.863e-05 max=0.2467 mean=0.03523
========== TEMP DEBUG: obs dict ==========
  obs['observation.images.primary'] shape=(3, 256, 256) dtype=torch.float32 min=0 max=1 mean=0.574
  obs['observation.images.wrist'] shape=(3, 256, 256) dtype=torch.float32 min=0 max=0.8196 mean=0.421
  obs['observation.images.empty_camera_0'] shape=(3, 224, 224) dtype=torch.float32 min=0 max=0 mean=0
  obs['observation.state.gripper'] shape=() dtype=torch.float32 min=0.2267 max=0.2267 mean=0.2267
  obs['observation.state.sensors_bota_ft_sensor'] shape=(6,) dtype=torch.float32 min=-0.1752 max=0.05842 mean=-0.04453
  obs['observation.state'] shape=(7,) dtype=torch.float32 min=-0.1752 max=0.2267 mean=-0.005789
  obs['task'] type=str value='stack the lego'
  obs['robot_type'] type=str value='franka'
========== TEMP DEBUG: obs_proc (post-preprocessor) ==========
  obs_proc['action'] type=NoneType value=None
  obs_proc['info'] type=dict value={}
  obs_proc['next.done'] type=bool value=False
  obs_proc['next.reward'] type=float value=0.0
  obs_proc['next.truncated'] type=bool value=False
  obs_proc['observation.images.empty_camera_0'] shape=(1, 3, 224, 224) dtype=torch.float32 min=0 max=0 mean=0
  obs_proc['observation.images.primary'] shape=(1, 3, 256, 256) dtype=torch.float32 min=0 max=1 mean=0.574
  obs_proc['observation.images.wrist'] shape=(1, 3, 256, 256) dtype=torch.float32 min=0 max=0.8196 mean=0.421
  obs_proc['observation.language.attention_mask'] shape=(1, 200) dtype=torch.bool min=0 max=1 mean=0.195
  obs_proc['observation.language.tokens'] shape=(1, 200) dtype=torch.int64 min=0 max=2.353e+05 mean=3.68e+04
  obs_proc['observation.state'] shape=(1, 7) dtype=torch.float32 min=-0.6622 max=0.9048 mean=0.06526
  obs_proc['observation.state.gripper'] shape=(1,) dtype=torch.float32 min=-0.6622 max=-0.6622 mean=-0.6622
  obs_proc['observation.state.sensors_bota_ft_sensor'] shape=(6,) dtype=torch.float32 min=-0.3937 max=0.9048 mean=0.1865
  obs_proc['task'] type=list value=['Task: stack the lego, State: 43 77 120 243 94 144 230;\nAction: ']
========== TEMP DEBUG: raw pred ==========
  pred shape=(1, 7) dtype=torch.float32 min=-1.58 max=0.02614 mean=-0.6611
========== TEMP DEBUG: post-postprocessor pred ==========
  pred shape=(1, 7) dtype=torch.float32 min=-0.0046 max=0.3605 mean=0.05029
========== END TEMP DEBUG ==========