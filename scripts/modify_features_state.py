#!/usr/bin/env python
"""Replace observation.state with gripper + ft_sensor concatenation.

Takes a LeRobot dataset and creates a new one where observation.state
is replaced by concat(observation.state.gripper[1], observation.state.sensors_bota_ft_sensor[6]) = shape [7].

Removes: observation.state (old 26-dim), observation.state.cartesian,
         observation.state.joints, observation.state.target

Usage:
    python scripts/modify_features_state.py \
        --dataset-path datasets/OliverHausdoerfer/stack_lego_simple_v1 \
        --output-path datasets/OliverHausdoerfer/stack_lego_simple_v1_new_state
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.dataset_tools import modify_features
from lerobot.datasets.io_utils import load_stats, write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import flatten_dict


def compute_new_state(row_dict, episode_index, frame_index):
    gripper = np.array(row_dict["observation.state.gripper"], dtype=np.float32).reshape(-1)
    ft_sensor = np.array(row_dict["observation.state.sensors_bota_ft_sensor"], dtype=np.float32).reshape(-1)
    return np.concatenate([gripper, ft_sensor])


def rename_feature_in_parquet_files(root: Path, old_name: str, new_name: str):
    """Rename a column in all parquet files under data/ and meta/episodes/."""
    for parquet_dir in ["data", "meta/episodes"]:
        parquet_files = sorted((root / parquet_dir).rglob("*.parquet"))
        for path in parquet_files:
            df = pd.read_parquet(path)
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
                df.to_parquet(path, index=False)


def rename_feature_in_json(root: Path, old_name: str, new_name: str):
    """Rename feature key in info.json and stats.json."""
    # info.json
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    if old_name in info["features"]:
        info["features"][new_name] = info["features"].pop(old_name)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    # stats.json
    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        if old_name in stats:
            stats[new_name] = stats.pop(old_name)
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)


def compute_per_episode_state_stats(output_path: Path, feature_spec: dict) -> dict[int, dict]:
    """Read data/*.parquet, return {ep_idx: compute_episode_stats output} for observation.state."""
    episode_arrays: dict[int, list[np.ndarray]] = {}
    for f in sorted((output_path / "data").rglob("*.parquet")):
        df = pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
        for ep_idx_raw in df["episode_index"].unique():
            ep_idx = int(ep_idx_raw)
            ep_df = df[df["episode_index"] == ep_idx].sort_values("frame_index")
            arr = np.stack([np.asarray(v, dtype=np.float32) for v in ep_df["observation.state"]])
            episode_arrays.setdefault(ep_idx, []).append(arr)

    per_episode_stats: dict[int, dict] = {}
    for ep_idx, chunks in episode_arrays.items():
        full = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
        per_episode_stats[ep_idx] = compute_episode_stats({"observation.state": full}, feature_spec)
    return per_episode_stats


def rewrite_episode_parquet_stats(
    output_path: Path,
    per_episode_stats: dict[int, dict],
    stale_prefixes: tuple[str, ...],
):
    """Drop stale stats columns and write new observation.state stats into meta/episodes/*.parquet."""
    flat_by_ep: dict[int, dict] = {
        ep: flatten_dict({"stats": {"observation.state": s["observation.state"]}})
        for ep, s in per_episode_stats.items()
    }
    stat_cols = list(next(iter(flat_by_ep.values())).keys())

    for ep_file in sorted((output_path / "meta/episodes").rglob("*.parquet")):
        df = pd.read_parquet(ep_file)
        drop_cols = [c for c in df.columns if c.startswith(stale_prefixes)]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        for col in stat_cols:
            df[col] = [flat_by_ep[int(ep)][col] for ep in df["episode_index"]]
        df.to_parquet(ep_file, index=False)


def main():
    parser = argparse.ArgumentParser(description="Replace observation.state with gripper + ft_sensor")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/OliverHausdoerfer/stack_lego_simple_v1",
        help="Path to the source dataset",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path for the output dataset (default: <dataset-path>_new_state)",
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
        output_path = str(dataset_path) + "_new_state"
    output_path = Path(output_path).resolve()

    repo_id = args.repo_id or dataset_path.name + "_new_state"

    print(f"Loading dataset from {dataset_path}")
    dataset = LeRobotDataset(repo_id=dataset_path.name, root=dataset_path)
    print(f"Dataset: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames")
    print(f"Existing features: {list(dataset.meta.features.keys())}")

    features_to_remove = [
        "observation.state",
        "observation.state.cartesian",
        "observation.state.joints",
        "observation.state.target",
    ]

    # Pass 1: single modify_features call -- add new state (temp name) and drop old state features in one data copy
    print(
        "\nPass 1: modify_features — add observation.state_new=[7] (gripper+ft_sensor), "
        f"remove {features_to_remove}"
    )
    modified = modify_features(
        dataset,
        add_features={
            "observation.state_new": (
                compute_new_state,
                {
                    "dtype": "float32",
                    "shape": (7,),
                    "names": [
                        "gripper",
                        "sensors_bota_ft_sensor_0",
                        "sensors_bota_ft_sensor_1",
                        "sensors_bota_ft_sensor_2",
                        "sensors_bota_ft_sensor_3",
                        "sensors_bota_ft_sensor_4",
                        "sensors_bota_ft_sensor_5",
                    ],
                },
            ),
        },
        remove_features=features_to_remove,
        output_dir=output_path,
        repo_id=repo_id,
    )
    print(f"Pass 1 done. Features: {list(modified.meta.features.keys())}")

    # Pass 2: rename observation.state_new -> observation.state, recompute stats, rewrite episode parquet
    print("\nPass 2a: rename observation.state_new -> observation.state (data parquet + info.json)")
    rename_feature_in_parquet_files(output_path, "observation.state_new", "observation.state")
    rename_feature_in_json(output_path, "observation.state_new", "observation.state")

    # Reload to get updated feature spec
    result = LeRobotDataset(repo_id=repo_id, root=output_path)
    feat_spec = {"observation.state": result.meta.features["observation.state"]}

    print("Pass 2b: computing per-episode stats for observation.state")
    per_episode_stats = compute_per_episode_state_stats(output_path, feat_spec)
    print(f"  Computed stats for {len(per_episode_stats)} episodes")

    print("Pass 2c: aggregating + writing meta/stats.json")
    aggregated = aggregate_stats(list(per_episode_stats.values()))
    existing = load_stats(output_path) or {}
    existing["observation.state"] = aggregated["observation.state"]
    write_stats(existing, output_path)

    print("Pass 2d: rewriting meta/episodes/*.parquet (drop stale stats cols, insert new)")
    stale_prefixes = (
        "stats/observation.state/",
        "stats/observation.state.cartesian/",
        "stats/observation.state.joints/",
        "stats/observation.state.target/",
    )
    rewrite_episode_parquet_stats(output_path, per_episode_stats, stale_prefixes)

    # Verify by reloading
    print("\nVerifying output dataset...")
    result = LeRobotDataset(repo_id=repo_id, root=output_path)
    print(f"Features: {list(result.meta.features.keys())}")
    print(f"observation.state shape: {result.meta.features['observation.state']['shape']}")
    print(f"Episodes: {result.meta.total_episodes}, Frames: {result.meta.total_frames}")

    state_stats = (result.meta.stats or {}).get("observation.state")
    if state_stats is None:
        print("WARNING: observation.state missing from stats")
    else:
        print(f"observation.state stats keys: {list(state_stats.keys())}")
        print(f"  mean: {state_stats['mean']}")
        print(f"  std:  {state_stats['std']}")

    sample = result[0]
    print(f"Sample observation.state: {sample['observation.state']}")
    print(f"\nDone! Output at: {output_path}")


if __name__ == "__main__":
    main()
