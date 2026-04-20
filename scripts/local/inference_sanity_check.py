#!/usr/bin/env python3
"""
Inference sanity check: compare ground truth vs predicted actions for 3 episodes.
Tests whether the policy learned something during training.

Usage (from repo root):
    python scripts/local/inference_sanity_check.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import numpy as np
import torch
import matplotlib.pyplot as plt

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

# ── Config ─────────────────────────────────────────────────────────────────────
CHECKPOINT = "OliverHausdoerfer/pi05_stack_lego_simple_20260418"
DATASET_REPO = "OliverHausdoerfer/stack_lego_simple_v1_filtered_new_state_fixed_stats"
DATASET_ROOT = "datasets/OliverHausdoerfer/stack_lego_simple_v1_filtered_new_state_fixed_stats"
EPISODES = [0, 1, 2]
TASK = "stack the lego"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
OUT_PATH = "inference_sanity_check.png"

# ── Load policy ────────────────────────────────────────────────────────────────
print(f"Loading policy from {CHECKPOINT} on {DEVICE} ...")
config = PreTrainedConfig.from_pretrained(CHECKPOINT)
config.compile_model = False  # skip compilation for faster startup

policy_cls = get_policy_class(config.type)
policy = policy_cls.from_pretrained(CHECKPOINT, config=config)
policy = policy.to(DEVICE)
policy.eval()

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=config,
    pretrained_path=CHECKPOINT,
    preprocessor_overrides={"device_processor": {"device": DEVICE}},
)
print("Policy ready.\n")

# ── Load dataset ───────────────────────────────────────────────────────────────
print(f"Loading dataset (episodes {EPISODES}) ...")
dataset = LeRobotDataset(DATASET_REPO, root=DATASET_ROOT, episodes=EPISODES)
print(f"Loaded {len(dataset)} frames.\n")

# empty camera required by policy but absent from dataset
EMPTY_CAM = torch.zeros(3, 224, 224)


def make_obs(sample: dict) -> dict:
    return {
        "observation.images.primary": sample["observation.images.primary"],
        "observation.images.wrist": sample["observation.images.wrist"],
        "observation.images.empty_camera_0": EMPTY_CAM.clone(),
        "observation.state.gripper": sample["observation.state.gripper"],
        "observation.state.sensors_bota_ft_sensor": sample["observation.state.sensors_bota_ft_sensor"],
        "observation.state": sample["observation.state"],
        "task": TASK,
        "robot_type": "franka",
    }


# ── Inference loop ─────────────────────────────────────────────────────────────
results = {}

for ep_idx in EPISODES:
    # v3: filter by episode_index column in hf_dataset
    ep_indices = [
        i for i, e in enumerate(dataset.hf_dataset["episode_index"]) if e == ep_idx
    ]
    n_frames = len(ep_indices)
    print(f"Episode {ep_idx}: {n_frames} frames ...")

    gt_list, pred_list = [], []
    policy.reset()

    for i, frame_idx in enumerate(ep_indices):
        sample = dataset[frame_idx]
        gt_list.append(sample["action"].numpy())

        obs = make_obs(sample)
        obs_proc = preprocessor(obs)

        with torch.inference_mode():
            pred = policy.select_action(obs_proc)

        pred = postprocessor(pred)
        pred_list.append(pred.squeeze(0).cpu().numpy())

        if i % 100 == 0:
            print(f"  {i}/{n_frames}")

    results[ep_idx] = {
        "gt": np.array(gt_list),    # [T, 7]
        "pred": np.array(pred_list),  # [T, 7]
    }
    print(f"  Done.\n")

# ── Plot ───────────────────────────────────────────────────────────────────────
n_dims = len(ACTION_NAMES)
n_eps = len(EPISODES)

fig, axes = plt.subplots(
    n_dims, n_eps,
    figsize=(5 * n_eps, 2.2 * n_dims),
    sharex=False,
    squeeze=False,
)
fig.suptitle("GT vs Predicted Actions — policy sanity check", fontsize=13, y=1.01)

for col, ep_idx in enumerate(EPISODES):
    gt = results[ep_idx]["gt"]
    pred = results[ep_idx]["pred"]
    T = gt.shape[0]
    t = np.arange(T)

    for row, dim_name in enumerate(ACTION_NAMES):
        ax = axes[row][col]
        ax.plot(t, gt[:, row],   color="#2196F3", lw=1.4, label="GT",        alpha=0.9)
        ax.plot(t, pred[:, row], color="#F44336", lw=1.4, label="Predicted",  alpha=0.85, linestyle="--")
        ax.set_ylabel(dim_name, fontsize=9)
        ax.grid(True, alpha=0.25)

        if row == 0:
            ax.set_title(f"Episode {ep_idx}", fontsize=10)
            ax.legend(fontsize=7, loc="upper right")
        if row == n_dims - 1:
            ax.set_xlabel("Step", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
plt.show()
