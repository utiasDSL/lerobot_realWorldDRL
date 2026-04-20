#!/usr/bin/env python
"""Filter zero-action frames from a LeRobot v3 dataset.

Removes frames where the robot is idle (no cartesian movement, no gripper movement)
UNLESS the force/torque sensor shows significant contact force (pressing against surface).

Episodes are preserved — removed frames simply shorten episodes. Episodes where ALL
frames are filtered out are dropped entirely.

Usage:
    # First check what would be filtered (dry run):
    python scripts/filter_zero_action_frames.py \
        --repo-id OliverHausdoerfer/stack_lego_simple_v1 \
        --root datasets/OliverHausdoerfer/stack_lego_simple_v1 \
        --dry-run

    # Then run the actual filtering:
    python scripts/filter_zero_action_frames.py \
        --repo-id OliverHausdoerfer/stack_lego_simple_v1 \
        --root datasets/OliverHausdoerfer/stack_lego_simple_v1 \
        --output-repo-id OliverHausdoerfer/stack_lego_simple_v1_filtered \
        --output-root datasets/OliverHausdoerfer/stack_lego_simple_v1_filtered
"""

import argparse
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_tools import (
    _keep_episodes_from_video_with_av,
    _write_parquet,
)
from lerobot.datasets.io_utils import write_info, write_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    DATA_DIR,
    DEFAULT_DATA_PATH,
    flatten_dict,
    update_chunk_file_indices,
)
from lerobot.datasets.video_utils import get_video_info
from lerobot.utils.constants import HF_LEROBOT_HOME

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class FilterConfig:
    cart_threshold: float = 1e-4
    gripper_threshold: float = 1e-3
    ft_threshold: float = 5.0
    min_episode_length: int = 2


@dataclass
class FilteredEpisode:
    """Describes one filtered episode (same identity as original, just shorter)."""

    old_episode_idx: int
    new_episode_idx: int
    keep_mask: np.ndarray  # boolean mask over original frames
    old_tasks: list
    old_n_frames: int
    new_n_frames: int  # sum of keep_mask


# ---------------------------------------------------------------------------
# Phase 1: classify frames
# ---------------------------------------------------------------------------


def classify_frames_for_episode(
    ep_data: pd.DataFrame, config: FilterConfig
) -> tuple[np.ndarray, dict]:
    """Return boolean keep-mask and per-frame diagnostics for one episode."""
    actions = np.stack(ep_data["action"].values)  # (N, 7)

    cart_norm = np.linalg.norm(actions[:, :6], axis=1)
    cart_moving = cart_norm > config.cart_threshold

    gripper_vals = actions[:, 6]
    gripper_delta = np.abs(np.diff(gripper_vals, prepend=gripper_vals[0]))
    gripper_moving = gripper_delta > config.gripper_threshold

    ft_sensor = np.stack(
        ep_data["observation.state.sensors_bota_ft_sensor"].values
    )  # (N, 6)
    ft_norm = np.linalg.norm(ft_sensor, axis=1)
    ft_significant = ft_norm > config.ft_threshold

    keep = cart_moving | gripper_moving | ft_significant

    diagnostics = {
        "cart_moving": int(cart_moving.sum()),
        "gripper_moving": int(gripper_moving.sum()),
        "ft_significant": int(ft_significant.sum()),
        "kept_total": int(keep.sum()),
    }
    return keep, diagnostics


def analyze_episodes(
    dataset: LeRobotDataset, all_data: pd.DataFrame, config: FilterConfig
) -> tuple[list[FilteredEpisode], dict]:
    """Analyze all episodes, return filtered episodes and statistics."""
    filtered_episodes: list[FilteredEpisode] = []
    new_ep_idx = 0
    stats = {
        "total_frames_before": 0,
        "total_frames_after": 0,
        "total_episodes_before": dataset.meta.total_episodes,
        "total_episodes_after": 0,
        "episodes_removed": 0,
        "per_episode": [],
    }

    for old_ep_idx in range(dataset.meta.total_episodes):
        ep_data = all_data[all_data["episode_index"] == old_ep_idx]
        ep_data = ep_data.sort_values("frame_index").reset_index(drop=True)
        n_frames = len(ep_data)
        stats["total_frames_before"] += n_frames

        keep_mask, diag = classify_frames_for_episode(ep_data, config)
        n_kept = int(keep_mask.sum())

        ep_tasks = dataset.meta.episodes[old_ep_idx]["tasks"]

        stats["per_episode"].append(
            {
                "old_ep": old_ep_idx,
                "frames_before": n_frames,
                "frames_after": n_kept,
                **diag,
            }
        )
        stats["total_frames_after"] += n_kept

        if n_kept < config.min_episode_length:
            stats["episodes_removed"] += 1
            continue

        filtered_episodes.append(
            FilteredEpisode(
                old_episode_idx=old_ep_idx,
                new_episode_idx=new_ep_idx,
                keep_mask=keep_mask,
                old_tasks=ep_tasks,
                old_n_frames=n_frames,
                new_n_frames=n_kept,
            )
        )
        new_ep_idx += 1

    stats["total_episodes_after"] = new_ep_idx
    return filtered_episodes, stats


def print_statistics(stats: dict) -> None:
    total_before = stats["total_frames_before"]
    total_after = stats["total_frames_after"]
    removed = total_before - total_after
    pct = removed / total_before * 100 if total_before > 0 else 0

    print("\n" + "=" * 60)
    print("FILTERING STATISTICS")
    print("=" * 60)
    print(f"Frames:   {total_before:>8} -> {total_after:>8}  ({removed} removed, {pct:.1f}%)")
    print(
        f"Episodes: {stats['total_episodes_before']:>8} -> {stats['total_episodes_after']:>8}"
    )
    print(f"  - episodes fully removed: {stats['episodes_removed']}")

    print("\nPer-episode breakdown:")
    print(
        f"  {'ep':>4} {'before':>7} {'after':>7} {'removed':>8} "
        f"{'cart_mv':>8} {'grip_mv':>8} {'ft_sig':>8}"
    )
    for ep in stats["per_episode"]:
        rm = ep["frames_before"] - ep["frames_after"]
        print(
            f"  {ep['old_ep']:>4} {ep['frames_before']:>7} {ep['frames_after']:>7} "
            f"{rm:>8} "
            f"{ep['cart_moving']:>8} {ep['gripper_moving']:>8} {ep['ft_significant']:>8}"
        )
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_all_parquet_data(dataset: LeRobotDataset) -> pd.DataFrame:
    data_dir = dataset.root / DATA_DIR
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")
    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True)


def mask_to_ranges(mask: np.ndarray) -> list[tuple[int, int]]:
    """Convert boolean mask to list of (start, end) half-open ranges of True values."""
    ranges = []
    in_run = False
    start = 0
    for i, k in enumerate(mask):
        if k and not in_run:
            start = i
            in_run = True
        elif not k and in_run:
            ranges.append((start, i))
            in_run = False
    if in_run:
        ranges.append((start, len(mask)))
    return ranges


# ---------------------------------------------------------------------------
# Phase 2: write filtered parquet data
# ---------------------------------------------------------------------------


def write_filtered_data(
    filtered_episodes: list[FilteredEpisode],
    all_data: pd.DataFrame,
    dataset: LeRobotDataset,
    new_meta: LeRobotDatasetMetadata,
    task_mapping: dict[int, int],
) -> dict[int, dict]:
    """Write filtered+reindexed parquet files. Returns per-episode data metadata."""
    episode_data_metadata: dict[int, dict] = {}
    global_index = 0
    chunk_idx, file_idx = 0, 0
    accumulated_dfs = []

    for fe in tqdm(filtered_episodes, desc="Building filtered data"):
        ep_data = all_data[all_data["episode_index"] == fe.old_episode_idx]
        ep_data = ep_data.sort_values("frame_index").reset_index(drop=True)

        # Apply keep mask
        sub_df = ep_data[fe.keep_mask].copy().reset_index(drop=True)
        n_frames = len(sub_df)

        # Reindex
        sub_df["episode_index"] = fe.new_episode_idx
        sub_df["frame_index"] = list(range(n_frames))
        sub_df["index"] = list(range(global_index, global_index + n_frames))
        sub_df["timestamp"] = [i / dataset.meta.fps for i in range(n_frames)]
        if task_mapping:
            sub_df["task_index"] = sub_df["task_index"].replace(task_mapping)

        episode_data_metadata[fe.new_episode_idx] = {
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": global_index,
            "dataset_to_index": global_index + n_frames,
        }
        global_index += n_frames
        accumulated_dfs.append(sub_df)

    # Write all accumulated data in one parquet file
    if accumulated_dfs:
        full_df = pd.concat(accumulated_dfs, ignore_index=True)
        dst_path = new_meta.root / DEFAULT_DATA_PATH.format(
            chunk_index=chunk_idx, file_index=file_idx
        )
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        _write_parquet(full_df, dst_path, new_meta)

    return episode_data_metadata


# ---------------------------------------------------------------------------
# Phase 3: re-encode videos
# ---------------------------------------------------------------------------


def reencode_filtered_videos(
    dataset: LeRobotDataset,
    new_meta: LeRobotDatasetMetadata,
    filtered_episodes: list[FilteredEpisode],
    vcodec: str,
    pix_fmt: str,
) -> dict[int, dict]:
    """Re-encode videos keeping only kept frames per episode.

    Returns dict mapping new_episode_idx -> video metadata.
    """
    video_metadata: dict[int, dict] = {fe.new_episode_idx: {} for fe in filtered_episodes}
    fps = dataset.meta.fps

    for video_key in dataset.meta.video_keys:
        logging.info(f"Re-encoding video: {video_key}")

        # Group episodes by source video file
        file_to_episodes: dict[tuple[int, int], list[FilteredEpisode]] = {}
        for fe in filtered_episodes:
            src_ep = dataset.meta.episodes[fe.old_episode_idx]
            src_chunk = src_ep[f"videos/{video_key}/chunk_index"]
            src_file = src_ep[f"videos/{video_key}/file_index"]
            key = (src_chunk, src_file)
            if key not in file_to_episodes:
                file_to_episodes[key] = []
            file_to_episodes[key].append(fe)

        dst_chunk_idx, dst_file_idx = 0, 0

        for (src_chunk, src_file), episodes_in_file in tqdm(
            sorted(file_to_episodes.items()),
            desc=f"  {video_key} video files",
        ):
            # Collect all frame ranges to keep from this source video file
            all_frame_ranges: list[tuple[int, int]] = []
            episode_frame_counts: list[tuple[FilteredEpisode, int]] = []

            for fe in episodes_in_file:
                src_ep = dataset.meta.episodes[fe.old_episode_idx]
                video_ep_start = round(
                    src_ep[f"videos/{video_key}/from_timestamp"] * fps
                )

                # Convert keep_mask to frame ranges (absolute within the video file)
                runs = mask_to_ranges(fe.keep_mask)
                ep_kept_frames = 0
                for run_start, run_end in runs:
                    abs_start = video_ep_start + run_start
                    abs_end = video_ep_start + run_end
                    all_frame_ranges.append((abs_start, abs_end))
                    ep_kept_frames += run_end - run_start

                episode_frame_counts.append((fe, ep_kept_frames))

            # Source and destination video paths
            src_video_path = dataset.root / dataset.meta.video_path.format(
                video_key=video_key, chunk_index=src_chunk, file_index=src_file
            )
            dst_video_path = new_meta.root / new_meta.video_path.format(
                video_key=video_key,
                chunk_index=dst_chunk_idx,
                file_index=dst_file_idx,
            )
            dst_video_path.parent.mkdir(parents=True, exist_ok=True)

            _keep_episodes_from_video_with_av(
                input_path=src_video_path,
                output_path=dst_video_path,
                episodes_to_keep=all_frame_ranges,
                fps=fps,
                vcodec=vcodec,
                pix_fmt=pix_fmt,
            )

            # Track per-episode timestamps in the output video
            cumulative_ts = 0.0
            for fe, n_kept_frames in episode_frame_counts:
                duration = n_kept_frames / fps
                video_metadata[fe.new_episode_idx][
                    f"videos/{video_key}/chunk_index"
                ] = dst_chunk_idx
                video_metadata[fe.new_episode_idx][
                    f"videos/{video_key}/file_index"
                ] = dst_file_idx
                video_metadata[fe.new_episode_idx][
                    f"videos/{video_key}/from_timestamp"
                ] = cumulative_ts
                video_metadata[fe.new_episode_idx][
                    f"videos/{video_key}/to_timestamp"
                ] = cumulative_ts + duration
                cumulative_ts += duration

            dst_chunk_idx, dst_file_idx = update_chunk_file_indices(
                dst_chunk_idx, dst_file_idx, new_meta.chunks_size
            )

    return video_metadata


# ---------------------------------------------------------------------------
# Phase 4: write episode metadata and stats
# ---------------------------------------------------------------------------


def compute_and_write_metadata(
    dataset: LeRobotDataset,
    new_meta: LeRobotDatasetMetadata,
    filtered_episodes: list[FilteredEpisode],
    all_data: pd.DataFrame,
    episode_data_metadata: dict[int, dict],
    video_metadata: dict[int, dict],
) -> None:
    """Compute per-episode stats and write episode metadata."""
    non_video_features = {
        k: v
        for k, v in dataset.meta.features.items()
        if v["dtype"] not in ["video", "image"]
    }

    for fe in tqdm(filtered_episodes, desc="Computing stats & writing metadata"):
        ep_data = all_data[all_data["episode_index"] == fe.old_episode_idx]
        ep_data = ep_data.sort_values("frame_index").reset_index(drop=True)
        sub_df = ep_data[fe.keep_mask].reset_index(drop=True)

        # Build episode_data dict for compute_episode_stats
        ep_buffer = {}
        for key in non_video_features:
            if key in sub_df.columns:
                vals = sub_df[key].values
                if hasattr(vals[0], "__len__"):
                    ep_buffer[key] = np.stack(vals)
                else:
                    ep_buffer[key] = np.array(vals)

        ep_stats = compute_episode_stats(ep_buffer, non_video_features)

        # Build episode metadata dict
        ep_meta = episode_data_metadata[fe.new_episode_idx].copy()
        if fe.new_episode_idx in video_metadata:
            ep_meta.update(video_metadata[fe.new_episode_idx])

        new_meta.save_episode(
            episode_index=fe.new_episode_idx,
            episode_length=fe.new_n_frames,
            episode_tasks=fe.old_tasks,
            episode_stats=ep_stats,
            episode_metadata=ep_meta,
        )

    new_meta._close_writer()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def create_filtered_dataset(
    dataset: LeRobotDataset,
    filtered_episodes: list[FilteredEpisode],
    all_data: pd.DataFrame,
    output_repo_id: str,
    output_dir: Path,
    vcodec: str,
    pix_fmt: str,
) -> LeRobotDataset:
    # Clean output dir if exists
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Create metadata
    new_meta = LeRobotDatasetMetadata.create(
        repo_id=output_repo_id,
        fps=dataset.meta.fps,
        features=dataset.meta.features,
        robot_type=dataset.meta.robot_type,
        root=output_dir,
        use_videos=len(dataset.meta.video_keys) > 0,
    )

    # Save tasks
    all_tasks = set()
    for fe in filtered_episodes:
        all_tasks.update(fe.old_tasks)
    new_meta.save_episode_tasks(list(all_tasks))

    # Build task mapping: old task_index -> new task_index
    task_mapping = {}
    if dataset.meta.tasks is not None:
        for old_task_idx in range(len(dataset.meta.tasks)):
            task_name = dataset.meta.tasks.iloc[old_task_idx].name
            new_task_idx = new_meta.get_task_index(task_name)
            if new_task_idx is not None:
                task_mapping[old_task_idx] = new_task_idx

    # Write filtered parquet data
    logging.info("Writing filtered parquet data...")
    episode_data_metadata = write_filtered_data(
        filtered_episodes, all_data, dataset, new_meta, task_mapping
    )

    # Re-encode videos
    video_metadata: dict[int, dict] = {}
    if dataset.meta.video_keys:
        logging.info("Re-encoding videos...")
        video_metadata = reencode_filtered_videos(
            dataset, new_meta, filtered_episodes, vcodec, pix_fmt
        )

    # Update video info from first output video
    if new_meta.video_keys:
        for vkey in new_meta.video_keys:
            video_path = new_meta.root / new_meta.video_path.format(
                video_key=vkey, chunk_index=0, file_index=0
            )
            if video_path.exists():
                new_meta.info["features"][vkey]["info"] = get_video_info(video_path)
        write_info(new_meta.info, new_meta.root)

    # Compute stats and write episode metadata
    logging.info("Computing statistics and writing episode metadata...")
    compute_and_write_metadata(
        dataset, new_meta, filtered_episodes, all_data, episode_data_metadata, video_metadata
    )

    # Load and return
    logging.info(f"Loading filtered dataset from {output_dir}")
    return LeRobotDataset(repo_id=output_repo_id, root=output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Filter zero-action frames from a LeRobot dataset"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Source dataset repo ID (e.g. OliverHausdoerfer/stack_lego_simple_v1)",
    )
    parser.add_argument("--root", type=str, default=None, help="Local root for source dataset")
    parser.add_argument(
        "--output-repo-id",
        type=str,
        default=None,
        help="Output dataset repo ID (default: {repo_id}_filtered)",
    )
    parser.add_argument("--output-root", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--cart-threshold",
        type=float,
        default=1e-4,
        help="Norm threshold for cartesian delta action (default: 1e-4)",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=1e-3,
        help="Threshold for gripper value change between frames (default: 1e-3)",
    )
    parser.add_argument(
        "--ft-threshold",
        type=float,
        default=5.0,
        help="Norm threshold for F/T sensor to count as pressing (default: 5.0)",
    )
    parser.add_argument(
        "--min-episode-length",
        type=int,
        default=2,
        help="Minimum frames for episode to be kept after filtering (default: 2)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print statistics, do not create output dataset",
    )
    parser.add_argument("--vcodec", type=str, default="libsvtav1", help="Video codec (default: libsvtav1)")
    parser.add_argument("--pix-fmt", type=str, default="yuv420p", help="Pixel format (default: yuv420p)")

    args = parser.parse_args()

    config = FilterConfig(
        cart_threshold=args.cart_threshold,
        gripper_threshold=args.gripper_threshold,
        ft_threshold=args.ft_threshold,
        min_episode_length=args.min_episode_length,
    )

    output_repo_id = args.output_repo_id or f"{args.repo_id}_filtered"
    output_dir = Path(args.output_root) if args.output_root else HF_LEROBOT_HOME / output_repo_id

    # Load source dataset
    logging.info(f"Loading dataset: {args.repo_id}")
    root = Path(args.root) if args.root else None
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=root,
        download_videos=not args.dry_run,
    )

    logging.info(
        f"Dataset: {dataset.meta.total_episodes} episodes, "
        f"{dataset.meta.total_frames} frames, {dataset.meta.fps} fps"
    )

    # Load all parquet data
    logging.info("Loading parquet data...")
    all_data = load_all_parquet_data(dataset)

    # Analyze episodes
    logging.info("Classifying frames...")
    filtered_episodes, filter_stats = analyze_episodes(dataset, all_data, config)

    print_statistics(filter_stats)

    if args.dry_run:
        logging.info("Dry run — no output dataset created.")
        return

    if not filtered_episodes:
        logging.error("All episodes were filtered out. Nothing to write.")
        return

    logging.info(f"Creating filtered dataset: {output_repo_id} at {output_dir}")
    new_dataset = create_filtered_dataset(
        dataset=dataset,
        filtered_episodes=filtered_episodes,
        all_data=all_data,
        output_repo_id=output_repo_id,
        output_dir=output_dir,
        vcodec=args.vcodec,
        pix_fmt=args.pix_fmt,
    )

    print(f"\nDone. Output dataset: {new_dataset.repo_id}")
    print(f"  Episodes: {new_dataset.meta.total_episodes}")
    print(f"  Frames:   {new_dataset.meta.total_frames}")
    print(f"  Root:     {new_dataset.root}")


if __name__ == "__main__":
    main()
