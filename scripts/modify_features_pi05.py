#!/usr/bin/env python
"""Add observation.state_pi05 feature to a LeRobot dataset.

Combines observation.state.joints (7) and observation.state.gripper (1)
into a single observation.state_pi05 feature of shape [8].

Usage:
    python scripts/adapt_dataset_features.py \
        --dataset-path datasets/continuallearning/real_0_put_bowl \
        --output-path datasets/continuallearning/real_0_put_bowl_pi05
"""

import argparse
from pathlib import Path

import numpy as np

from lerobot.datasets.dataset_tools import add_features, remove_feature
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def compute_state_pi05(row_dict, episode_index, frame_index):
    joints = np.array(row_dict["observation.state.joints"], dtype=np.float32)
    gripper = np.array(row_dict["observation.state.gripper"], dtype=np.float32).reshape(-1)
    return np.concatenate([joints, gripper])


def main():
    parser = argparse.ArgumentParser(description="Add observation.state_pi05 feature to dataset")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/continuallearning/real_0_put_bowl",
        help="Path to the source dataset",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path for the output dataset (default: <dataset-path>_pi05)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Repo ID for the output dataset",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path).resolve()
    output_path = args.output_path
    if output_path is None:
        output_path = str(dataset_path) + "_pi05"
    output_path = Path(output_path).resolve()

    repo_id = args.repo_id or dataset_path.name + "_pi05"

    print(f"Loading dataset from {dataset_path}")
    dataset = LeRobotDataset(repo_id=dataset_path.name, root=dataset_path)
    print(f"Dataset: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames")
    print(f"Existing features: {list(dataset.meta.features.keys())}")

    print(f"\nAdding observation.state_pi05 (joints[7] + gripper[1] = [8])")
    print(f"Output: {output_path}")

    new_dataset = add_features(
        dataset,
        features={
            "observation.state_pi05": (
                compute_state_pi05,
                {
                    "dtype": "float32",
                    "shape": (8,),
                    "names": [
                        "joint_0",
                        "joint_1",
                        "joint_2",
                        "joint_3",
                        "joint_4",
                        "joint_5",
                        "joint_6",
                        "gripper",
                    ],
                },
            ),
        },
        output_dir=f"{output_path}_temp",
        repo_id=repo_id,
    )

    print(f"\nFirst pass done. Features: {list(new_dataset.meta.features.keys())}")

    # Second pass: remove old state features
    features_to_remove = [
        "observation.state.cartesian",
        "observation.state.gripper",
        "observation.state.joints",
        "observation.state.target",
        "observation.state",
    ]
    print(f"\nRemoving old features: {features_to_remove}")
    final_dataset = remove_feature(
        new_dataset,
        feature_names=features_to_remove,
        output_dir=output_path,
        repo_id=repo_id,
    )

    print(f"\nDone! Final features: {list(final_dataset.meta.features.keys())}")
    print(f"Output dataset at: {output_path}")


if __name__ == "__main__":
    main()
