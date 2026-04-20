#!/usr/bin/env python
"""Download, inspect, and augment a LeRobot dataset with quantile stats for pi0.5."""

from pprint import pprint

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.io_utils import write_stats
from lerobot.scripts.augment_dataset_quantile_stats import (
    compute_quantile_stats_for_dataset,
    has_quantile_stats,
)
from lerobot.scripts.lerobot_dataset_viz import visualize_dataset
from lerobot.utils.utils import init_logging

# run with :
# conda activate lerobot_realWorldDRL
# python scripts/download_and_inspect_dataset.py


REPO_ID = "HuggingFaceVLA/libero"
ROOT = f"datasets/{REPO_ID}"
EPISODE_INDEX = 0


def main():
    init_logging()

    # Download full dataset locally
    print(f"Downloading  dataset: {REPO_ID} to {ROOT}/...")
    dataset = LeRobotDataset(REPO_ID, root=ROOT)

    # Print metadata summary
    print("\n" + "=" * 60)
    print("DATASET METADATA")
    print("=" * 60)
    print(dataset.meta)
    print(f"\nTotal episodes: {dataset.meta.total_episodes}")
    print(f"Total frames: {dataset.meta.total_frames}")
    print(f"FPS: {dataset.meta.fps}")
    print(f"Robot type: {dataset.meta.robot_type}")
    print(f"Camera keys: {dataset.meta.camera_keys}")
    print(f"\nSelected episodes: {dataset.episodes}")
    print(f"Selected frames: {dataset.num_frames}")

    print("\nFeatures:")
    pprint(dataset.meta.features)

    print("\nTasks:")
    print(dataset.meta.tasks)

    # Print sample frame shapes
    print("\n" + "=" * 60)
    print("SAMPLE FRAME (index 0)")
    print("=" * 60)
    sample = dataset[0]
    for key, val in sample.items():
        if hasattr(val, "shape"):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key}: {val}")

    # Compute quantile stats for pi0.5 compatibility
    print("\n" + "=" * 60)
    print("COMPUTING QUANTILE STATS")
    print("=" * 60)
    if has_quantile_stats(dataset.meta.stats):
        print("Dataset already has quantile stats. Skipping.")
    else:
        print("Computing quantile statistics (required for pi0.5)...")
        new_stats = compute_quantile_stats_for_dataset(dataset)
        dataset.meta.stats = new_stats
        write_stats(new_stats, dataset.meta.root)
        print(f"Quantile stats saved to {dataset.meta.root}")

    # Launch Rerun visualizer
    print("\n" + "=" * 60)
    print("LAUNCHING RERUN VISUALIZER")
    print("=" * 60)
    visualize_dataset(dataset, episode_index=EPISODE_INDEX, batch_size=32, num_workers=0)


if __name__ == "__main__":
    main()
