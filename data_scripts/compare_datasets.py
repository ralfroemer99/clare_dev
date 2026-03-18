"""
Compare structure and values of two LeRobot datasets.
Usage:
    python data_scripts/compare_datasets.py \
        --ds_a continuallearning/libero_10_image_task_0 \
        --ds_b continuallearning/libero_spatial_image_task_0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "lerobot_lsy/src")
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def load_info(ds: LeRobotDataset) -> dict:
    return json.loads((Path(ds.root) / "meta" / "info.json").read_text())


def compare_info(info_a: dict, info_b: dict, name_a: str, name_b: str):
    print("\n=== info.json ===")
    keys_to_skip = {"total_episodes", "total_frames", "splits"}  # expected to differ
    ok = True
    for key in set(info_a) | set(info_b):
        if key in keys_to_skip:
            continue
        va, vb = info_a.get(key), info_b.get(key)
        if va != vb:
            print(f"  MISMATCH [{key}]: {name_a}={va!r}  {name_b}={vb!r}")
            ok = False
    if ok:
        print("  OK — codebase_version, fps, features match")

    print(f"\n  {name_a}: {info_a['total_episodes']} episodes, {info_a['total_frames']} frames")
    print(f"  {name_b}: {info_b['total_episodes']} episodes, {info_b['total_frames']} frames")


def load_all_parquet(ds: LeRobotDataset) -> pd.DataFrame:
    root = Path(ds.root)
    files = sorted(root.glob("data/**/*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def compare_parquet_schema(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str):
    print("\n=== Parquet schema ===")
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    only_a = cols_a - cols_b
    only_b = cols_b - cols_a
    if only_a:
        print(f"  Columns only in {name_a}: {only_a}")
    if only_b:
        print(f"  Columns only in {name_b}: {only_b}")

    ok = True
    for col in sorted(cols_a & cols_b):
        ta, tb = df_a[col].dtype, df_b[col].dtype
        if ta != tb:
            print(f"  DTYPE MISMATCH [{col}]: {name_a}={ta}  {name_b}={tb}")
            ok = False
    if ok and not only_a and not only_b:
        print("  OK — same columns and dtypes")


def compare_episode_lengths(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str):
    print("\n=== Episode lengths ===")
    def ep_lengths(df):
        return df.groupby("episode_index").size()

    la = ep_lengths(df_a)
    lb = ep_lengths(df_b)
    print(f"  {name_a}: min={la.min()}, max={la.max()}, mean={la.mean():.1f}")
    print(f"  {name_b}: min={lb.min()}, max={lb.max()}, mean={lb.mean():.1f}")


def compare_value_stats(df_a: pd.DataFrame, df_b: pd.DataFrame, name_a: str, name_b: str):
    print("\n=== Value statistics (action & observation.state) ===")
    numeric_cols = ["action", "observation.state"]
    for col in numeric_cols:
        if col not in df_a.columns or col not in df_b.columns:
            continue
        print(f"\n  [{col}]")
        # Each cell is a list/array; stack into 2D
        arr_a = np.stack(df_a[col].values)
        arr_b = np.stack(df_b[col].values)
        print(f"    shape  — {name_a}: {arr_a.shape}  {name_b}: {arr_b.shape}")
        print(f"    min    — {name_a}: {arr_a.min():.4f}  {name_b}: {arr_b.min():.4f}")
        print(f"    max    — {name_a}: {arr_a.max():.4f}  {name_b}: {arr_b.max():.4f}")
        print(f"    mean   — {name_a}: {arr_a.mean():.4f}  {name_b}: {arr_b.mean():.4f}")
        print(f"    dtype  — {name_a}: {arr_a.dtype}  {name_b}: {arr_b.dtype}")

    # Image spot-check: dtype and value range
    import io
    from PIL import Image

    img_col = "observation.images.image"
    if img_col in df_a.columns and img_col in df_b.columns:
        print(f"\n  [{img_col}] (first frame only)")
        def decode_image(cell):
            # Parquet stores images as {'bytes': b'...', 'path': '...'} dicts
            raw = cell["bytes"] if isinstance(cell, dict) else cell
            return np.array(Image.open(io.BytesIO(raw)))

        img_a = decode_image(df_a[img_col].iloc[0])
        img_b = decode_image(df_b[img_col].iloc[0])
        print(f"    shape  — {name_a}: {img_a.shape}  {name_b}: {img_b.shape}")
        print(f"    dtype  — {name_a}: {img_a.dtype}  {name_b}: {img_b.dtype}")
        print(f"    range  — {name_a}: [{img_a.min()}, {img_a.max()}]  {name_b}: [{img_b.min()}, {img_b.max()}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_a", required=True, help="repo_id of reference dataset (libero_10_image_task_0)")
    parser.add_argument("--ds_b", required=True, help="repo_id of new dataset (libero_spatial_image_task_0)")
    args = parser.parse_args()

    print(f"Loading {args.ds_a} ...")
    ds_a = LeRobotDataset(repo_id=args.ds_a)
    print(f"Loading {args.ds_b} ...")
    ds_b = LeRobotDataset(repo_id=args.ds_b)

    name_a = args.ds_a.split("/")[-1]
    name_b = args.ds_b.split("/")[-1]

    info_a = load_info(ds_a)
    info_b = load_info(ds_b)
    compare_info(info_a, info_b, name_a, name_b)

    print("\nLoading parquet files ...")
    df_a = load_all_parquet(ds_a)
    df_b = load_all_parquet(ds_b)

    compare_parquet_schema(df_a, df_b, name_a, name_b)
    compare_episode_lengths(df_a, df_b, name_a, name_b)
    compare_value_stats(df_a, df_b, name_a, name_b)


if __name__ == "__main__":
    main()
