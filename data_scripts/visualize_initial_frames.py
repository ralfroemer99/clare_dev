"""
Overlay all initial frames from both cameras of a LeRobot v3.0 dataset
to visualize the distribution of initial object positions.
"""

import argparse
from pathlib import Path

import av
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage

# HSV ranges for colour detection (PIL HSV scale: H 0-255, S 0-255, V 0-255).
# PIL maps 0-360° hue to 0-255, so divide degree values by 1.41.
COLOR_RANGES = {
    "orange": dict(h_lo=7, h_hi=21, s_lo=100, v_lo=80, rgb=(1.0, 0.5, 0.0)),
    "yellow": dict(h_lo=21, h_hi=42, s_lo=80, v_lo=80, rgb=(1.0, 0.9, 0.0)),
    "gray":   dict(h_lo=0, h_hi=255, s_lo=0, s_hi=50, v_lo=60, v_hi=180, rgb=(0.6, 0.6, 0.6)),
}


def color_mask(frame: np.ndarray, color: str) -> np.ndarray:
    """Return a boolean mask of pixels matching the given colour."""
    cfg = COLOR_RANGES[color]
    hsv = np.array(PILImage.fromarray(frame).convert("HSV"))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (h >= cfg["h_lo"]) & (h <= cfg["h_hi"]) & (s >= cfg["s_lo"]) & (v >= cfg["v_lo"])
    if "s_hi" in cfg:
        mask = mask & (s <= cfg["s_hi"])
    if "v_hi" in cfg:
        mask = mask & (v <= cfg["v_hi"])
    return mask


def get_first_frames(dataset_dir: Path, camera_key: str) -> list[np.ndarray]:
    """Return the first frame of each episode for a given camera.

    Uses the v3.0 episodes metadata to find the correct video file for each
    episode and decodes frame 0 from it.
    """
    video_dir = dataset_dir / "videos" / camera_key / "chunk-000"

    # Read episodes metadata for the video file mapping
    ep_meta_dir = dataset_dir / "meta" / "episodes" / "chunk-000"
    ep_dfs = []
    for f in sorted(ep_meta_dir.glob("*.parquet")):
        try:
            ep_dfs.append(pd.read_parquet(f))
        except Exception:
            print(f"    Skipping corrupt metadata: {f.name}")
    ep_df = pd.concat(ep_dfs).sort_values("episode_index")

    file_idx_col = f"videos/{camera_key}/file_index"

    frames = []
    for _, row in ep_df.iterrows():
        video_file_idx = int(row[file_idx_col])
        video_path = video_dir / f"file-{video_file_idx:03d}.mp4"
        with av.open(str(video_path)) as container:
            frame = next(container.decode(video=0))
            frames.append(np.array(frame.to_image()))
    return frames


def make_mean(frames: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(frames, axis=0).astype(np.float32)
    return np.clip(stack.mean(axis=0), 0, 255).astype(np.uint8)


def make_heatmap(frames: list[np.ndarray], color: str) -> np.ndarray:
    heatmap = np.zeros(frames[0].shape[:2], dtype=np.float32)
    for frame in frames:
        heatmap += color_mask(frame, color).astype(np.float32)
    return heatmap


DATASETS = [
    ("continuallearning/real_0_put_bowl_filtered", "orange"),
    ("continuallearning/real_1_stack_bowls_filtered", "orange"),
    ("continuallearning/real_2_put_moka_pot_filtered", None),
    ("continuallearning/real_3_close_drawer_filtered", None),
    ("continuallearning/real_4_put_lego_into_drawer_filtered", "yellow"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=str(Path.home() / ".cache/huggingface/lerobot"))
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    cameras = ["observation.images.primary", "observation.images.wrist"]
    camera_labels = ["Third-person camera", "Wrist camera"]

    for dataset_id, obj_color in DATASETS:
        dataset_dir = Path(args.cache_dir) / dataset_id
        short_name = dataset_id.split("/")[-1]
        print(f"\n=== {dataset_id} (color={obj_color}) ===")

        ep_meta_dir = dataset_dir / "meta" / "episodes" / "chunk-000"
        ep_dfs = []
        for f in sorted(ep_meta_dir.glob("*.parquet")):
            try:
                ep_dfs.append(pd.read_parquet(f))
            except Exception:
                pass
        ep_df = pd.concat(ep_dfs)
        n_episodes = len(ep_df)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        mode = f"object colour: {obj_color}" if obj_color else "mean overlay only"
        fig.suptitle(f"Initial frames — {short_name}\n({n_episodes} episodes, {mode})", fontsize=13)

        for ax, camera_key, label in zip(axes, cameras, camera_labels):
            print(f"  {camera_key}…", end=" ", flush=True)
            frames = get_first_frames(dataset_dir, camera_key)
            print(f"{len(frames)} frames")
            background = make_mean(frames)
            ax.imshow(background)

            if obj_color is not None:
                heatmap = make_heatmap(frames, obj_color)
                norm = heatmap / max(heatmap.max(), 1)
                rgb = COLOR_RANGES[obj_color]["rgb"]
                cmap = mcolors.LinearSegmentedColormap.from_list(
                    obj_color, [(*rgb, 0), (*rgb, 1)],
                )
                overlay_rgba = cmap(norm)
                overlay_rgba[..., 3] = np.where(heatmap > 0, 0.3 + 0.7 * norm, 0.0)
                ax.imshow(overlay_rgba)

            ax.set_title(label, fontsize=11)
            ax.axis("off")

        plt.tight_layout()
        out_path = Path(args.output_dir) / f"initial_frames_{short_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
