#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert a LeRobot dataset from codebase version 3.0 back to 2.1 and (optionally) push it
to the Hugging Face Hub under a new repo name.

What this script does:
- Downloads the v3.0 dataset from the Hub (unless a local copy is given).
- Splits concatenated data parquet files back into per-episode files.
- Splits concatenated video files back into per-episode files (using ffmpeg).
- Converts parquet metadata (episodes, tasks, episode stats) back to JSONL format.
- Restores total_chunks / total_videos and sets codebase_version = "v2.1" in info.json.
- Pushes the result to the Hub as <output-repo-id> (default: <namespace>/<name>_lerobot21).

Usage:

Download from Hub, convert, and push back:
```bash
python src/lerobot/datasets/v30/convert_dataset_v30_to_v21.py \\
    --repo-id=lerobot/pusht
```

Use a local copy and skip the Hub push:
```bash
python src/lerobot/datasets/v30/convert_dataset_v30_to_v21.py \\
    --repo-id=lerobot/pusht \\
    --root=/path/to/datasets \\
    --push-to-hub=false
```

Specify a custom output repo:
```bash
python src/lerobot/datasets/v30/convert_dataset_v30_to_v21.py \\
    --repo-id=lerobot/pusht \\
    --output-repo-id=myorg/pusht_v21
```
"""

import argparse
import io
import json
import logging
import math
import shutil
import subprocess
from pathlib import Path

import av
import jsonlines
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
from huggingface_hub import HfApi, snapshot_download

from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_VIDEO_PATH,
    LEGACY_EPISODES_PATH,
    LEGACY_EPISODES_STATS_PATH,
    LEGACY_TASKS_PATH,
    load_info,
    unflatten_dict,
    write_info,
)
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.utils import init_logging

V21 = "v2.1"
V30 = "v3.0"

# v2.1 path templates (variable name must be episode_chunk, matching original v2.1 convention)
LEGACY_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
LEGACY_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


# ──────────────────────────── helpers ────────────────────────────


def _default_output_repo_id(repo_id: str) -> str:
    """Turn 'namespace/name' into 'namespace/name_lerobot21'."""
    namespace, name = repo_id.split("/", 1)
    return f"{namespace}/{name}_lerobot21"


def validate_local_dataset_version(local_path: Path) -> None:
    info = load_info(local_path)
    version = info.get("codebase_version", "unknown")
    if version != V30:
        raise ValueError(
            f"Dataset at {local_path} has version '{version}', expected '{V30}'. "
            "This script only converts v3.0 → v2.1."
        )


def load_v30_episodes(root: Path) -> pd.DataFrame:
    paths = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode parquet files found under {root / 'meta' / 'episodes'}")
    return (
        pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        .sort_values("episode_index")
        .reset_index(drop=True)
    )


def get_video_keys(root: Path) -> list[str]:
    info = load_info(root)
    return [k for k, ft in info["features"].items() if ft["dtype"] == "video"]


# ──────────────────────────── metadata ────────────────────────────


def convert_tasks(root: Path, new_root: Path) -> None:
    logging.info("Converting tasks → tasks.jsonl")
    # v3.0 tasks.parquet: task string is the index, task_index is a column
    tasks_df = pd.read_parquet(root / "meta" / "tasks.parquet")
    fpath = new_root / LEGACY_TASKS_PATH
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(fpath, "w") as writer:
        for task_str, row in tasks_df.iterrows():
            writer.write({"task_index": int(row["task_index"]), "task": str(task_str)})


def convert_episodes(episodes_df: pd.DataFrame, new_root: Path) -> None:
    logging.info("Converting episodes → episodes.jsonl")
    fpath = new_root / LEGACY_EPISODES_PATH
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(fpath, "w") as writer:
        for _, row in episodes_df.iterrows():
            tasks = row["tasks"]
            if not isinstance(tasks, list):
                tasks = list(tasks)
            writer.write({
                "episode_index": int(row["episode_index"]),
                "tasks": tasks,
                "length": int(row["length"]),
            })


def _to_python(obj):
    """Recursively convert numpy scalars/arrays (including nested object arrays) to plain Python."""
    if isinstance(obj, np.ndarray):
        return _to_python(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, list):
        return [_to_python(x) for x in obj]
    return obj


def convert_episodes_stats(episodes_df: pd.DataFrame, new_root: Path) -> None:
    stats_cols = [c for c in episodes_df.columns if c.startswith("stats/")]
    if not stats_cols:
        logging.warning("No stats/* columns in episodes metadata – skipping episodes_stats.jsonl.")
        return

    logging.info("Converting episode stats → episodes_stats.jsonl")
    fpath = new_root / LEGACY_EPISODES_STATS_PATH
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(fpath, "w") as writer:
        for _, row in episodes_df.iterrows():
            flat = {
                k.removeprefix("stats/"): _to_python(row[k])
                for k in stats_cols
            }
            stats = unflatten_dict(flat)
            writer.write({
                "episode_index": int(row["episode_index"]),
                "stats": stats,
            })


# ──────────────────────────── data parquet ────────────────────────────


def split_data_files(root: Path, new_root: Path) -> None:
    logging.info("Splitting data parquet files…")
    # Scan all parquet files in the data directory rather than relying on path templates,
    # because some datasets use 1-based file numbering while the metadata stores 0-based indices.
    data_files = sorted((root / "data").glob("*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    for src_path in tqdm.tqdm(data_files, desc="split data files"):
        src_df = pd.read_parquet(src_path)
        for ep_idx in src_df["episode_index"].unique():
            ep_idx = int(ep_idx)
            ep_df = src_df[src_df["episode_index"] == ep_idx]
            dest = new_root / LEGACY_DATA_PATH.format(
                episode_chunk=ep_idx // DEFAULT_CHUNK_SIZE, episode_index=ep_idx
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            ep_df.to_parquet(dest, index=False)


# ──────────────────────────── videos ────────────────────────────


def _ffmpeg_extract_segment(src: Path, dest: Path, from_ts: float, to_ts: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ss", str(from_ts),
        "-t", str(to_ts - from_ts),
        "-c", "copy",
        "-movflags", "faststart",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed extracting {src} [{from_ts}–{to_ts}]:\n{result.stderr}")


def split_videos(root: Path, new_root: Path, episodes_df: pd.DataFrame) -> None:
    video_keys = get_video_keys(root)
    if not video_keys:
        return

    logging.info(f"Splitting videos for cameras: {video_keys}")
    for video_key in sorted(video_keys):
        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        from_col = f"videos/{video_key}/from_timestamp"
        to_col = f"videos/{video_key}/to_timestamp"

        for (chunk_idx, file_idx), group in tqdm.tqdm(
            episodes_df.groupby([chunk_col, file_col]),
            desc=f"split {video_key}",
        ):
            src = root / DEFAULT_VIDEO_PATH.format(
                video_key=video_key, chunk_index=int(chunk_idx), file_index=int(file_idx)
            )
            for _, ep_row in group.iterrows():
                ep_idx = int(ep_row["episode_index"])
                dest = new_root / LEGACY_VIDEO_PATH.format(
                    episode_chunk=ep_idx // DEFAULT_CHUNK_SIZE,
                    video_key=video_key,
                    episode_index=ep_idx,
                )
                _ffmpeg_extract_segment(src, dest, float(ep_row[from_col]), float(ep_row[to_col]))


# ──────────────────────────── embed video frames ────────────────────────────


def _decode_video_frames_as_jpeg(video_path: Path) -> list[dict]:
    """Decode all frames from a video file, returning each as a JPEG bytes dict."""
    frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(video=0):
            buf = io.BytesIO()
            frame.to_image().save(buf, format="JPEG", quality=95)
            frames.append({"bytes": buf.getvalue(), "path": None})
    return frames


def embed_video_frames_in_parquet(new_root: Path, video_keys: list[str]) -> None:
    """Add image columns to each episode parquet by decoding the corresponding video files."""
    if not video_keys:
        return

    logging.info("Embedding video frames into parquet files…")
    ep_parquet_paths = sorted((new_root / "data").glob("*/*.parquet"))

    for parquet_path in tqdm.tqdm(ep_parquet_paths, desc="embed frames"):
        df = pd.read_parquet(parquet_path)
        ep_idx = int(df["episode_index"].iloc[0])
        ep_chunk = ep_idx // DEFAULT_CHUNK_SIZE

        col_data: dict[str, list] = {}
        for video_key in video_keys:
            video_path = new_root / LEGACY_VIDEO_PATH.format(
                episode_chunk=ep_chunk, video_key=video_key, episode_index=ep_idx
            )
            frames = _decode_video_frames_as_jpeg(video_path)
            if len(frames) != len(df):
                raise ValueError(
                    f"Frame count mismatch for {video_path}: "
                    f"{len(frames)} video frames vs {len(df)} parquet rows"
                )
            col_data[video_key] = frames

        for key, frames in col_data.items():
            df[key] = frames

        # Write back with proper HF Image feature schema
        image_cols = list(col_data.keys())
        schema_fields = []
        for field in pa.Schema.from_pandas(df):
            if field.name in image_cols:
                schema_fields.append(
                    pa.field(field.name, pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())]))
                )
            else:
                schema_fields.append(field)
        table = pa.Table.from_pandas(df, schema=pa.schema(schema_fields))
        pq.write_table(table, parquet_path)


# ──────────────────────────── info.json ────────────────────────────


def convert_info(root: Path, new_root: Path, episodes_df: pd.DataFrame, embed_images: bool) -> None:
    info = load_info(root)
    num_episodes = len(episodes_df)
    video_keys = get_video_keys(root)

    info["codebase_version"] = V21
    info["total_chunks"] = math.ceil(num_episodes / DEFAULT_CHUNK_SIZE) if num_episodes else 1
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)
    info["data_path"] = LEGACY_DATA_PATH

    if embed_images and video_keys:
        # Change video features to image features; no video files in the output
        for key in video_keys:
            info["features"][key]["dtype"] = "image"
        info["total_videos"] = 0
        info["video_path"] = None
    else:
        info["total_videos"] = num_episodes * len(video_keys)
        info["video_path"] = LEGACY_VIDEO_PATH if video_keys else None

    # v2.1 does not store fps inside each feature dict
    for ft in info["features"].values():
        ft.pop("fps", None)

    write_info(info, new_root)


# ──────────────────────────── README / dataset card ────────────────────────────


def write_readme(local_dir: Path) -> None:
    """Write a README.md matching the standard LeRobot v2.1 dataset card format."""
    info = load_info(local_dir)
    info_json = json.dumps(info, indent=4, ensure_ascii=False)

    content = (
        "---\n"
        "license: apache-2.0\n"
        "task_categories:\n"
        "- robotics\n"
        "tags:\n"
        "- LeRobot\n"
        "configs:\n"
        "- config_name: default\n"
        "  data_files: data/*/*.parquet\n"
        "---\n"
        "\n"
        "This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).\n"
        "\n"
        "## Dataset Description\n"
        "\n"
        "- **Homepage:** [More Information Needed]\n"
        "- **Paper:** [More Information Needed]\n"
        "- **License:** apache-2.0\n"
        "\n"
        "## Dataset Structure\n"
        "\n"
        "[meta/info.json](meta/info.json):\n"
        "```json\n"
        f"{info_json}\n"
        "```\n"
    )
    (local_dir / "README.md").write_text(content)


# ──────────────────────────── hub push ────────────────────────────


def push_to_hub(local_dir: Path, output_repo_id: str) -> None:
    logging.info(f"Pushing v2.1 dataset to Hub as '{output_repo_id}'…")
    write_readme(local_dir)
    api = HfApi()
    api.create_repo(repo_id=output_repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=output_repo_id,
        repo_type="dataset",
    )
    api.create_tag(output_repo_id, tag=V21, repo_type="dataset", exist_ok=True)
    logging.info(f"Done. Dataset available at https://huggingface.co/datasets/{output_repo_id}")


# ──────────────────────────── top-level ────────────────────────────


def convert_dataset(
    repo_id: str,
    output_repo_id: str | None = None,
    root: str | Path | None = None,
    push_to_hub_flag: bool = True,
    embed_images: bool = True,
) -> None:
    if output_repo_id is None:
        output_repo_id = _default_output_repo_id(repo_id)

    # Resolve local source directory
    if root is None:
        src_dir = HF_LEROBOT_HOME / repo_id
    else:
        src_dir = Path(root) / repo_id

    # Download from hub if needed
    if not src_dir.exists():
        logging.info(f"Downloading v3.0 dataset '{repo_id}' from the Hub…")
        snapshot_download(repo_id, repo_type="dataset", revision=V30, local_dir=src_dir)

    validate_local_dataset_version(src_dir)

    video_keys = get_video_keys(src_dir)

    # Build output directory next to source
    out_dir = src_dir.parent / f"{src_dir.name}_lerobot21"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    logging.info(f"Converting {src_dir} → {out_dir}")

    episodes_df = load_v30_episodes(src_dir)

    convert_info(src_dir, out_dir, episodes_df, embed_images=embed_images)
    convert_tasks(src_dir, out_dir)
    convert_episodes(episodes_df, out_dir)
    convert_episodes_stats(episodes_df, out_dir)
    split_data_files(src_dir, out_dir)
    split_videos(src_dir, out_dir, episodes_df)
    if embed_images:
        embed_video_frames_in_parquet(out_dir, video_keys)

    logging.info(f"Conversion complete. v2.1 dataset written to {out_dir}")

    if push_to_hub_flag:
        push_to_hub(out_dir, output_repo_id)


# ──────────────────────────── CLI ────────────────────────────


if __name__ == "__main__":
    init_logging()
    parser = argparse.ArgumentParser(
        description="Convert a LeRobot v3.0 dataset to v2.1 and push it to the Hub as NAME_lerobot21."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hub repo of the v3.0 source dataset, e.g. 'lerobot/pusht'.",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        default=None,
        help=(
            "Hub repo to push the converted v2.1 dataset to. "
            "Defaults to '<namespace>/<name>_lerobot21'."
        ),
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "Local directory used for caching / reading the source dataset. "
            "If omitted, HF_LEROBOT_HOME/<repo-id> is used. "
            "The converted dataset is written alongside the source."
        ),
    )
    parser.add_argument(
        "--push-to-hub",
        type=lambda s: s.lower() == "true",
        default=True,
        help="Push the converted dataset to the Hub (default: true).",
    )
    parser.add_argument(
        "--embed-images",
        type=lambda s: s.lower() == "true",
        default=True,
        help=(
            "Decode video frames and embed them as image columns in the parquet files "
            "so the HF Dataset Viewer can display them (default: true). "
            "Set to false to keep video-only format."
        ),
    )

    args = parser.parse_args()
    convert_dataset(
        repo_id=args.repo_id,
        output_repo_id=args.output_repo_id,
        root=args.root,
        push_to_hub_flag=args.push_to_hub,
        embed_images=args.embed_images,
    )
