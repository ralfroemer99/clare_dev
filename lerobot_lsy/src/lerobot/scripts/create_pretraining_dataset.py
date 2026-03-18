"""Merge multiple LeRobot datasets into one pre-training dataset."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import list_datasets
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.video_utils import decode_video_frames

# Each entry is (repo_id, task_string).
# task_string overrides the language command stored in the source dataset.
MERGE_DATASETS: list[tuple[str, str]] = [
    ("LSY-lab/drawer_v1",                      "Put the lego block in the drawer"), #
    ("LSY-lab/drawer_v2",                      "Put the lego block in the drawer"), #
    ("LSY-lab/drawer_without_tact_v2",         "Put the lego block in the drawer"), #
    ("LSY-lab/plate_v1",                       "Put the bowl on the plate"), #
    ("LSY-lab/sort_mug_v1",                    "Put the red mug on the red plate and the green mug on the green plate."), #
    ("LSY-lab/stack_cake_v1",                  "Stack the white block onto the blue block"), #
    ("LSY-lab/stack_cake_v2",                  "Stack the white block onto the blue block"), #
    ("local/stack_cake_v2_delta",               "Stack the white block onto the blue block"), # converted from stack_cake_v2_absolute_actions
    ("local/stack_cake_v3_delta",               "Stack the white block onto the blue block"), # converted from stack_cake_v3
    ("LSY-lab/stack_v1",                       "Stack the small lego block on top of the big lego block"), #
    ("LSY-lab/stack_v2",                       "Stack the small lego block on top of the big lego block"), #
    ("LSY-lab/stack_v3",                       "Stack the small lego block on top of the big lego block"), #
    ("LSY-lab/stack_without_ft_tact_v2",       "Stack the small lego block on top of the big lego block"), #
    ("LSY-lab/stack_without_ft_tact_v4",       "Stack the small lego block on top of the big lego block"), #
    ("LSY-lab/stack_without_tact_v4",          "Stack the small lego block on top of the big lego block"), #
    ("LSY-lab/triage_v2",                      "Sort the Lego blocks into the correct boxes."), #
]

# Datasets added for pretraining_v2 (None = use per-episode task from source)
MERGE_DATASETS_V2: list[tuple[str, str | None]] = MERGE_DATASETS + [
    ("local/droid_2000_delta",                 None),  # per-episode language instructions from DROID
]

MERGE_REFERENCE = "continuallearning/real_0_put_bowl"

AUTO_KEYS = {"index", "episode_index", "frame_index", "timestamp", "task_index"}

MATCH_ORGS = ["LSY-lab", "continuallearning", "ralfroemer"]


def _feature_compatible(candidate_feat: dict, ref_feat: dict, key: str, contains: bool) -> bool:
    """Return True if a single feature is compatible.

    In contains-mode:
    - observation.state: accept a larger shape (superset of state dims).
    - video features: ignore the 'info' field (codec metadata may differ).
    - all other features: require exact equality.
    """
    if candidate_feat == ref_feat:
        return True
    if contains:
        if key == "observation.state":
            # Same dtype, candidate shape must be >= reference shape (more state dims is fine)
            return (
                candidate_feat.get("dtype") == ref_feat.get("dtype")
                and candidate_feat.get("shape", (0,))[0] >= ref_feat.get("shape", (0,))[0]
            )
        if ref_feat.get("dtype") == "video":
            # Ignore codec/encoding metadata — only dtype and shape must match
            return (
                candidate_feat.get("dtype") == ref_feat.get("dtype")
                and candidate_feat.get("shape") == ref_feat.get("shape")
            )
    return False


def _features_match(candidate: dict, reference: dict, contains: bool) -> bool:
    """Return True if *candidate* satisfies the match criterion vs *reference*.

    contains=False: exact equality.
    contains=True:  candidate must have every key/value from reference (may have extras).
                    observation.state may be larger than the reference.
    """
    if contains:
        return all(
            k in candidate and _feature_compatible(candidate[k], reference[k], k, contains)
            for k in reference
        )
    return candidate == reference


def list_hub_datasets(
    orgs: list[str],
    match_features_of: str | None = None,
    contains: bool = False,
) -> list[str]:
    """Print all datasets published by *orgs* on the HuggingFace Hub.

    If *match_features_of* is given, fetch metadata for every dataset and mark
    which ones satisfy the match criterion (no full download).
    contains=True relaxes the check to: dataset has at least all reference features.
    """
    repo_ids = sorted(ds.id for org in orgs for ds in list_datasets(author=org))
    orgs_str = ", ".join(f"'{o}'" for o in orgs)
    print(f"Datasets on {orgs_str} ({len(repo_ids)} total):\n")

    if match_features_of is None:
        for repo_id in repo_ids:
            print(f"  {repo_id}")
        return repo_ids

    mode = "contains all features of" if contains else "exactly matches features of"
    print(f"Checking which datasets {mode}: {match_features_of}\n")
    reference_features = load_meta(match_features_of).features

    matching: list[tuple[str, int, int]] = []   # (repo_id, episodes, frames)
    differing: dict[str, dict] = {}             # repo_id -> features (for non-matching datasets)
    for repo_id in repo_ids:
        if repo_id == match_features_of:
            print(f"  {repo_id:<55} (reference)")
            meta = load_meta(repo_id)
            matching.append((repo_id, meta.total_episodes, meta.total_frames))
            continue
        try:
            meta = load_meta(repo_id)
            same = _features_match(meta.features, reference_features, contains)
        except Exception:
            print(f"  {repo_id:<55} ERROR (not a LeRobot dataset or missing meta)")
            continue
        if same:
            matching.append((repo_id, meta.total_episodes, meta.total_frames))
        else:
            differing[repo_id] = meta.features
        label = "MATCH" if same else "different"
        print(f"  {repo_id:<55} {label}")

    # --- matching datasets: episodes/timesteps table ---
    total_episodes = sum(ep for _, ep, _ in matching)
    total_frames = sum(fr for _, _, fr in matching)

    print(f"\n{len(matching)} dataset(s) match:\n")
    print(f"  {'Dataset':<55} {'Episodes':>10} {'Timesteps':>12}")
    print(f"  {'-'*55} {'-'*10} {'-'*12}")
    for repo_id, episodes, frames in matching:
        print(f"  {repo_id:<55} {episodes:>10,} {frames:>12,}")
    print(f"  {'TOTAL':<55} {total_episodes:>10,} {total_frames:>12,}")

    # --- feature breakdown for differing datasets ---
    if differing:
        print(f"\nFeature differences (vs. '{match_features_of}'):\n")
        for repo_id, features in differing.items():
            only_in_ref = sorted(k for k in reference_features if k not in features)
            only_in_other = sorted(k for k in features if k not in reference_features)
            changed = sorted(
                k for k in reference_features
                if k in features and features[k] != reference_features[k]
            )
            print(f"  {repo_id}")
            if only_in_ref:
                print(f"    Missing features:  {', '.join(only_in_ref)}")
            if not contains and only_in_other:
                print(f"    Extra features:    {', '.join(only_in_other)}")
            if changed:
                for k in changed:
                    ref = reference_features[k]
                    oth = features[k]
                    all_subkeys = set(ref) | set(oth)
                    subdiffs = {sk for sk in all_subkeys if ref.get(sk) != oth.get(sk)}
                    for sk in sorted(subdiffs):
                        print(f"    Changed '{k}.{sk}': {ref.get(sk)} -> {oth.get(sk)}")
            if not only_in_ref and (contains or not only_in_other) and not changed:
                print("    (no feature-level differences found)")

    return [repo_id for repo_id, _, _ in matching]


def load_meta(repo_id: str) -> LeRobotDatasetMetadata:
    """Fetch only the meta/ folder for *repo_id* and return the metadata object."""
    return LeRobotDatasetMetadata(repo_id=repo_id)


def load_features(repo_id: str) -> dict:
    """Fetch only the meta/ folder for *repo_id* and return its features dict."""
    return load_meta(repo_id).features


def show_features(repo_id: str) -> None:
    """Print the features of a single dataset (metadata-only, no video/parquet download)."""
    print(f"Features for '{repo_id}':\n")
    features = load_features(repo_id)
    for key in sorted(features):
        feat = features[key]
        dtype = feat.get("dtype", "?")
        shape = feat.get("shape", "?")
        names = feat.get("names")
        suffix = f"  names={names}" if names else ""
        print(f"  {key:<45} {dtype}  {str(shape):<20}{suffix}")


def compare_dataset_features(repo_ids: list[str]) -> bool:
    """Fetch metadata for each dataset and compare their features.

    Only downloads meta/info.json — no videos or parquet files.
    Prints a detailed report. Returns True if all datasets share identical features.
    """
    print("Fetching dataset metadata (meta/ only)...\n")
    datasets: dict[str, dict] = {}
    for repo_id in repo_ids:
        print(f"  {repo_id} ...")
        datasets[repo_id] = load_features(repo_id)

    all_keys: set[str] = set()
    for features in datasets.values():
        all_keys.update(features.keys())

    all_keys = {k for k in all_keys if "tactile" not in k}

    reference_features = datasets[repo_ids[0]]

    col_w = 25
    print(f"\n{'Feature':<45} ", end="")
    for repo_id in repo_ids:
        print(f"{repo_id.split('/')[-1]:<{col_w}} ", end="")
    print()
    print("-" * (45 + (col_w + 1) * len(repo_ids)))

    for key in sorted(all_keys):
        row_match = all(
            key in datasets[r] and datasets[r][key] == reference_features.get(key)
            for r in repo_ids
        )
        print(f"{key:<45} ", end="")
        for repo_id in repo_ids:
            feat = datasets[repo_id].get(key)
            cell = "MISSING" if feat is None else f"{feat.get('dtype', '?')} {feat.get('shape', '?')}"
            print(f"{cell:<{col_w}} ", end="")
        print("" if row_match else " <-- MISMATCH")

    print()


def _build_state_index_map(src_names: list[str], ref_names: list[str]) -> list[int] | None:
    """Return indices into src_names for each name in ref_names, or None if they are identical."""
    if src_names == ref_names:
        return None
    src_idx = {name: i for i, name in enumerate(src_names)}
    missing = [n for n in ref_names if n not in src_idx]
    if missing:
        raise ValueError(f"observation.state names missing in source: {missing}")
    return [src_idx[n] for n in ref_names]


def convert_absolute_to_delta(source_repo_id: str, target_repo_id: str) -> None:
    """Convert a dataset with absolute actions to one with delta actions.

    For each frame: delta_action = absolute_action - observation.state[:7]
    observation.state[:7] = [x, y, z, roll, pitch, yaw, gripper] (current EEF pose).

    The converted dataset is stored locally in HF_LEROBOT_HOME / target_repo_id.
    All other features (videos, state, etc.) are copied unchanged.
    """
    from lerobot.constants import HF_LEROBOT_HOME

    source = LeRobotDataset(repo_id=source_repo_id, video_backend="pyav")
    src_features = source.meta.features

    # Verify observation.state[:7] matches action names
    state_names = src_features.get("observation.state", {}).get("names", [])
    action_names = src_features.get("action", {}).get("names", [])
    assert state_names[:len(action_names)] == action_names, (
        f"observation.state[:7] names {state_names[:7]} don't match action names {action_names}"
    )

    target_root = HF_LEROBOT_HOME / target_repo_id
    if target_root.exists():
        shutil.rmtree(target_root, ignore_errors=True)
        if target_root.exists():
            os.system(f"rm -rf '{target_root}'")

    target = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=source.meta.fps,
        robot_type=source.meta.robot_type,
        features=src_features,
        use_videos=True,
    )

    video_keys = [k for k in src_features if src_features[k].get("dtype") == "video"]
    scalar_keys = [k for k in src_features if k not in AUTO_KEYS and src_features[k].get("dtype") not in ("image", "video")]
    n_action_dims = len(action_names)

    ep_bar = tqdm(total=source.meta.total_episodes, desc="episodes", unit="ep")
    for ep_idx in range(source.meta.total_episodes):
        ep_start = int(source.episode_data_index["from"][ep_idx].item())
        ep_end   = int(source.episode_data_index["to"][ep_idx].item())
        n_frames = ep_end - ep_start

        ep_rows = source.hf_dataset.select(range(ep_start, ep_end))
        timestamps = [float(ep_rows[i]["timestamp"]) for i in range(n_frames)]

        decoded_videos: dict[str, list[np.ndarray]] = {}
        skip_episode = False
        for vid_key in video_keys:
            video_path = source.root / source.meta.get_video_file_path(ep_idx, vid_key)
            try:
                frames_tensor = decode_video_frames(video_path, timestamps, source.tolerance_s, "pyav")
            except Exception as e:
                print(f"\n  WARNING: skipping ep {ep_idx} ({type(e).__name__})")
                skip_episode = True
                break
            decoded_videos[vid_key] = [
                (frames_tensor[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                for i in range(n_frames)
            ]

        if skip_episode:
            ep_bar.update(1)
            continue

        for i in range(n_frames):
            frame_dict = {}
            for vid_key in video_keys:
                frame_dict[vid_key] = decoded_videos[vid_key][i]
            for key in scalar_keys:
                value = ep_rows[i][key]
                v = value.numpy() if hasattr(value, "numpy") else np.array(value)
                if key == "action":
                    state = ep_rows[i]["observation.state"]
                    state_np = state.numpy() if hasattr(state, "numpy") else np.array(state)
                    # Only convert Cartesian/rotation dims (x,y,z,roll,pitch,yaw) to delta.
                    # Gripper (last dim) stays absolute, matching the reference dataset convention.
                    delta = v.copy()
                    delta[:6] = v[:6] - state_np[:6]
                    v = delta
                expected_shape = src_features[key].get("shape")
                if isinstance(v, np.ndarray) and expected_shape and v.shape != tuple(expected_shape):
                    v = v.reshape(expected_shape)
                frame_dict[key] = v
            task = ep_rows[i]["task"] if "task" in ep_rows.column_names else ""
            target.add_frame(frame_dict, task=task)

        devnull = os.open(os.devnull, os.O_WRONLY)
        stderr_fd = sys.stderr.fileno()
        saved_stderr = os.dup(stderr_fd)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        try:
            target.save_episode()
        finally:
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stderr)

        ep_bar.update(1)

    ep_bar.close()
    print(f"\nConverted '{source_repo_id}' -> '{target_repo_id}'")
    print(f"  {target.meta.total_episodes} episodes, {target.meta.total_frames} frames")
    print(f"  Stored at: {target_root}")


def convert_droid_to_pretraining(
    source_repo_id: str,
    target_repo_id: str,
    episodes: list[int] | None = None,
    resume: bool = False,
) -> None:
    """Convert a subset of cadene/droid_1.0.1 into the pretraining_v1 feature structure.

    - Only episodes with a non-empty language_instruction are included.
    - action.original (absolute Cartesian) is converted to delta: action[:6] - state_cartesian[:6].
    - Gripper stays absolute.
    - exterior_1_left -> observation.images.primary  (resized 180x320 -> 256x256)
    - wrist_left      -> observation.images.wrist    (resized 180x320 -> 256x256)
    - observation.state: [x,y,z,roll,pitch,yaw,gripper, j0..j6, cart] (20 dims, target=cart)
    - Pass resume=True to continue an interrupted conversion (done episodes tracked in droid_progress.json).
    """
    import cv2
    import pyarrow.parquet as pq
    import glob
    from lerobot.constants import HF_LEROBOT_HOME

    # Maps target feature key -> DROID video_key (used in path: videos/chunk-XXX/{video_key}/episode_XXXXXX.mp4)
    DROID_CAMERAS = {
        "observation.images.primary": "observation.images.exterior_1_left",
        "observation.images.wrist":   "observation.images.wrist_left",
    }
    TARGET_IMG_SIZE = (256, 256)  # (H, W)
    STATE_DIM = 20  # matches pretraining_v1

    # Load reference features from pretraining_v1
    ref_meta = load_meta(MERGE_REFERENCE)
    target_features = ref_meta.features

    # Find cached parquet files
    cache_root = Path(
        os.path.expanduser("~/.cache/huggingface/lerobot")
    ) / source_repo_id
    parquet_dirs = sorted((cache_root / "data").glob("chunk-*"))
    all_parquets = sorted(f for d in parquet_dirs for f in d.glob("episode_*.parquet"))
    if episodes is not None:
        ep_set = set(episodes)
        all_parquets = [p for p in all_parquets if int(p.stem.split("_")[1]) in ep_set]
    print(f"Found {len(all_parquets)} parquet files for {source_repo_id}")

    target_root = HF_LEROBOT_HOME / target_repo_id
    progress_file = target_root / "droid_progress.json"

    # Load already-done episode indices if resuming
    done_ep_indices: set[int] = set()
    if resume and target_root.exists() and progress_file.exists():
        with open(progress_file) as f:
            done_ep_indices = set(json.load(f))
        print(f"Resuming: {len(done_ep_indices)} episodes already done, skipping them.")
        target = _make_writable_dataset(target_repo_id, target_root)
    else:
        if target_root.exists():
            shutil.rmtree(target_root, ignore_errors=True)
            if target_root.exists():
                os.system(f"rm -rf '{target_root}'")
        target = LeRobotDataset.create(
            repo_id=target_repo_id,
            fps=ref_meta.fps,
            robot_type=ref_meta.robot_type,
            features=target_features,
            use_videos=True,
        )

    skipped_no_lang = 0
    skipped_no_video = 0
    n_episodes = 0

    ep_bar = tqdm(all_parquets, desc="episodes", unit="ep")
    for parquet_path in ep_bar:
        ep_idx = int(parquet_path.stem.split("_")[1])

        if ep_idx in done_ep_indices:
            continue

        # Read parquet
        table = pq.read_table(parquet_path, columns=[
            "language_instruction", "language_instruction_2",
            "action.original",
            "observation.state.cartesian_position",
            "observation.state.gripper_position",
            "observation.state.joint_position",
            "timestamp",
        ])
        df = table.to_pandas()

        # Skip episodes without language
        lang = str(df["language_instruction"].iloc[0]).strip()
        if not lang or lang in ("None", "nan", ""):
            lang = str(df.get("language_instruction_2", [""])[0]).strip()
        if not lang or lang in ("None", "nan", ""):
            skipped_no_lang += 1
            continue

        # Load video frames for both cameras
        video_frames: dict[str, list[np.ndarray]] = {}
        skip_ep = False
        for tgt_key, cam_name in DROID_CAMERAS.items():
            video_path = cache_root / "videos" / f"chunk-{ep_idx // 1000:03d}" / cam_name / f"episode_{ep_idx:06d}.mp4"
            # cam_name is now the full feature key, e.g. observation.images.exterior_1_left
            if not video_path.exists():
                skipped_no_video += 1
                skip_ep = True
                break
            timestamps = df["timestamp"].tolist()
            try:
                frames_tensor = decode_video_frames(video_path, timestamps, 1e-4, "pyav")
            except Exception as e:
                print(f"\n  WARNING: skipping ep {ep_idx} ({type(e).__name__})")
                skip_ep = True
                break
            frames = []
            for i in range(len(df)):
                frame = (frames_tensor[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                frame = cv2.resize(frame, (TARGET_IMG_SIZE[1], TARGET_IMG_SIZE[0]))
                frames.append(frame)
            video_frames[tgt_key] = frames

        if skip_ep:
            continue

        # Write frames
        for i in range(len(df)):
            cart = np.array(df["observation.state.cartesian_position"].iloc[i], dtype=np.float32)  # (6,)
            grip = np.array(df["observation.state.gripper_position"].iloc[i], dtype=np.float32).reshape(1)  # (1,)
            joints = np.array(df["observation.state.joint_position"].iloc[i], dtype=np.float32)  # (7,)
            state = np.concatenate([cart, grip, joints, cart])  # (20,): [..., target=cart]

            action_abs = np.array(df["action.original"].iloc[i], dtype=np.float32)  # (7,)
            action_delta = action_abs.copy()
            action_delta[:6] = action_abs[:6] - cart[:6]  # delta Cartesian, gripper stays absolute

            frame_dict = {
                "action": action_delta,
                "observation.state":          state,
                "observation.state.cartesian": cart,
                "observation.state.gripper":   grip,
                "observation.state.joints":    joints,
                "observation.state.target":    cart,
                "observation.images.primary": video_frames["observation.images.primary"][i],
                "observation.images.wrist":   video_frames["observation.images.wrist"][i],
            }
            target.add_frame(frame_dict, task=lang)

        devnull = os.open(os.devnull, os.O_WRONLY)
        stderr_fd = sys.stderr.fileno()
        saved_stderr = os.dup(stderr_fd)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        try:
            target.save_episode()
        finally:
            os.dup2(saved_stderr, stderr_fd)
            os.close(saved_stderr)

        n_episodes += 1
        done_ep_indices.add(ep_idx)
        with open(progress_file, "w") as f:
            json.dump(sorted(done_ep_indices), f)

    ep_bar.close()
    print(f"\nConverted '{source_repo_id}' -> '{target_repo_id}'")
    print(f"  {n_episodes} episodes written ({skipped_no_lang} skipped: no language, {skipped_no_video} skipped: no video)")
    print(f"  {target.meta.total_frames} frames")
    print(f"  Stored at: {target_root}")


def _make_writable_dataset(repo_id: str, root) -> "LeRobotDataset":
    """Reconstruct an existing local LeRobotDataset in write mode (for resuming a merge)."""
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    obj = LeRobotDataset.__new__(LeRobotDataset)
    obj.meta = meta
    obj.repo_id = meta.repo_id
    obj.root = meta.root
    obj.revision = None
    obj.tolerance_s = 1e-4
    obj.image_writer = None
    obj.episode_buffer = obj.create_episode_buffer()
    obj.episodes = None
    obj.hf_dataset = obj.create_hf_dataset()
    obj.image_transforms = None
    obj.delta_timestamps = None
    obj.delta_indices = None
    obj.episode_data_index = None
    obj.video_backend = "pyav"
    return obj


def merge_datasets(
    sources: list[tuple[str, str]],
    reference_repo_id: str,
    target_repo_id: str,
    push: bool = False,
    private: bool = True,
    resume: bool = False,
):
    """Merge *sources* into one dataset whose feature structure matches *reference_repo_id*.

    *sources* is a list of (repo_id, task_string) pairs. The task_string overrides whatever
    language command is stored in the source dataset.

    Pass resume=True to continue an interrupted merge: the existing target is kept, already-
    processed sources are skipped (tracked in merge_progress.json), and the in-progress source
    resumes from where it left off based on episode counts.

    For each source frame:
    - Only features present in the reference are written.
    - observation.state is re-ordered/subselected by name so extra sensor dims are dropped.
    - Video features are accepted regardless of codec metadata differences.
    - Episodes with corrupted/truncated video are skipped with a warning.
    """
    import json
    from lerobot.constants import HF_LEROBOT_HOME

    ref_meta = load_meta(reference_repo_id)
    target_features = ref_meta.features
    ref_state_names = target_features.get("observation.state", {}).get("names") or []

    target_root = HF_LEROBOT_HOME / target_repo_id
    progress_file = target_root / "merge_progress.json"

    if resume and target_root.exists():
        target = _make_writable_dataset(target_repo_id, target_root)
        already_done_eps = target.meta.total_episodes
        done_sources: set[str] = set()
        if progress_file.exists():
            with open(progress_file) as f:
                done_sources = set(json.load(f).get("done", []))
        print(f"Resuming: target already has {already_done_eps} episodes; {len(done_sources)} source(s) fully done.")
    else:
        if target_root.exists():
            shutil.rmtree(target_root, ignore_errors=True)
            if target_root.exists():
                os.system(f"rm -rf '{target_root}'")
        target = LeRobotDataset.create(
            repo_id=target_repo_id,
            fps=ref_meta.fps,
            robot_type=ref_meta.robot_type,
            features=target_features,
            use_videos=True,
        )
        already_done_eps = 0
        done_sources = set()

    video_keys = [k for k in target_features if target_features[k].get("dtype") == "video"]
    scalar_keys = [k for k in target_features if k not in AUTO_KEYS and target_features[k].get("dtype") not in ("image", "video")]

    total_src_episodes = 0
    total_src_frames = 0
    cumulative_eps = 0

    for source_repo_id, task_string in sources:
        source = LeRobotDataset(repo_id=source_repo_id, video_backend="pyav")
        n_src_eps = source.meta.total_episodes

        if source_repo_id in done_sources:
            print(f"\n{source_repo_id}: already merged, skipping")
            total_src_episodes += n_src_eps
            total_src_frames += source.meta.total_frames
            cumulative_eps += n_src_eps
            continue

        if n_src_eps == 0:
            print(f"{source_repo_id}: 0 episodes, skipping")
            continue

        # When resuming, determine which episode to start from within this source
        start_ep = max(0, already_done_eps - cumulative_eps)
        cumulative_eps += n_src_eps

        src_state_names = source.meta.features.get("observation.state", {}).get("names") or []
        state_index_map = _build_state_index_map(src_state_names, ref_state_names)

        resume_note = f" (resuming from ep {start_ep})" if start_ep > 0 else ""
        task_label = "(per-episode)" if task_string is None else f"'{task_string}'"
        print(f"\n{source_repo_id}: {n_src_eps} ep, {source.meta.total_frames} frames  task={task_label}{resume_note}")
        total_src_episodes += n_src_eps
        total_src_frames += source.meta.total_frames

        ep_bar = tqdm(total=n_src_eps - start_ep, desc="  episodes", unit="ep", leave=False)

        for ep_idx in range(start_ep, n_src_eps):
            ep_start = int(source.episode_data_index["from"][ep_idx].item())
            ep_end = int(source.episode_data_index["to"][ep_idx].item())
            n_frames = ep_end - ep_start

            ep_rows = source.hf_dataset.select(range(ep_start, ep_end))
            timestamps = [float(ep_rows[i]["timestamp"]) for i in range(n_frames)]

            # Decode all video frames for this episode in one sequential pass
            decoded_videos: dict[str, list[np.ndarray]] = {}
            skip_episode = False
            for vid_key in video_keys:
                video_path = source.root / source.meta.get_video_file_path(ep_idx, vid_key)
                try:
                    frames_tensor = decode_video_frames(video_path, timestamps, source.tolerance_s, "pyav")
                except Exception as e:
                    print(f"\n  WARNING: skipping ep {ep_idx} of {source_repo_id} ({type(e).__name__}: truncated/corrupt video)")
                    skip_episode = True
                    break
                decoded_videos[vid_key] = [
                    (frames_tensor[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    for i in range(n_frames)
                ]

            if skip_episode:
                ep_bar.update(1)
                continue

            for i in range(n_frames):
                frame_dict = {}
                for vid_key in video_keys:
                    frame_dict[vid_key] = decoded_videos[vid_key][i]
                for key in scalar_keys:
                    value = ep_rows[i][key]
                    v = value.numpy() if hasattr(value, "numpy") else value
                    if key == "observation.state" and state_index_map is not None:
                        v = v[state_index_map]
                    else:
                        expected_shape = target_features[key].get("shape")
                        if isinstance(v, np.ndarray) and expected_shape and v.shape != tuple(expected_shape):
                            v = v.reshape(expected_shape)
                    frame_dict[key] = v
                ep_task = task_string
                if ep_task is None:
                    ep_task_idx = int(ep_rows[0]["task_index"])
                    ep_task = source.meta.tasks[ep_task_idx]
                target.add_frame(frame_dict, task=ep_task)

            # Save episode (suppress SVT-AV1 encoder spam on stderr)
            devnull = os.open(os.devnull, os.O_WRONLY)
            stderr_fd = sys.stderr.fileno()
            saved_stderr = os.dup(stderr_fd)
            os.dup2(devnull, stderr_fd)
            os.close(devnull)
            try:
                target.save_episode()
            finally:
                os.dup2(saved_stderr, stderr_fd)
                os.close(saved_stderr)

            ep_bar.update(1)

        ep_bar.close()

        # Mark source as fully done and persist progress
        done_sources.add(source_repo_id)
        with open(progress_file, "w") as f:
            json.dump({"done": list(done_sources)}, f)

    print(f"\nDone.")
    print(f"  Sources:  {len(sources)} datasets, {total_src_episodes} episodes, {total_src_frames} frames")
    print(f"  Target:   '{target_repo_id}', {target.meta.total_episodes} episodes, {target.meta.total_frames} frames")

    if push:
        target.push_to_hub(private=private)
        print(f"  Pushed to hub: {target_repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LeRobot pre-training dataset utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list: show all datasets for one or more orgs
    list_parser = subparsers.add_parser("list", help="List all datasets for one or more HuggingFace orgs")
    list_parser.add_argument(
        "--org",
        nargs="+",
        default=["LSY-lab"],
        metavar="ORG",
        help="HuggingFace org/user name(s) (default: LSY-lab)",
    )
    list_parser.add_argument(
        "--match",
        default=None,
        metavar="REPO_ID",
        help=(
            "If given, check which datasets share identical features with this repo (metadata only). "
            f"Searches across {MATCH_ORGS} by default unless --org is explicitly set."
        ),
    )
    list_parser.add_argument(
        "--contains",
        action="store_true",
        help="Relax the match to: dataset contains at least all features of --match (extra features are allowed).",
    )

    # features: show features of a single dataset (no full download)
    feat_parser = subparsers.add_parser("features", help="Show features of a single dataset (metadata only)")
    feat_parser.add_argument("repo_id", help="Dataset repo ID, e.g. LSY-lab/plate_v1")

    # compare: compare features across multiple datasets (no full download)
    cmp_parser = subparsers.add_parser("compare", help="Compare features across datasets (metadata only)")
    cmp_parser.add_argument(
        "--datasets",
        nargs="+",
        default=[repo_id for repo_id, _ in MERGE_DATASETS],
        help="Repo IDs to compare (default: MERGE_DATASETS list in this file)",
    )

    # convert: re-encode a dataset with transformed actions
    conv_parser = subparsers.add_parser("convert", help="Convert absolute actions to delta actions (stored locally)")
    conv_parser.add_argument("--source", required=True, help="Source repo ID, e.g. LSY-lab/stack_cake_v2_absolute_actions")
    conv_parser.add_argument("--target", required=True, help="Target local repo ID, e.g. local/stack_cake_v2_delta")

    # merge: combine multiple datasets into one with a reference feature structure
    merge_parser = subparsers.add_parser("merge", help="Merge multiple datasets into one pre-training dataset")
    merge_parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Source repo IDs to merge (default: MERGE_DATASETS list in this file). Task strings come from MERGE_DATASETS.",
    )
    merge_parser.add_argument(
        "--reference",
        default=MERGE_REFERENCE,
        help=f"Repo ID whose feature structure the output will follow (default: {MERGE_REFERENCE})",
    )
    merge_parser.add_argument("--target", required=True, help="Target repo ID for the merged dataset")
    merge_parser.add_argument("--push", action="store_true", help="Push merged dataset to HuggingFace Hub")
    merge_parser.add_argument("--public", action="store_true", help="Make pushed dataset public (default: private)")
    merge_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted merge: keep the existing target, skip already-done sources (tracked in merge_progress.json).",
    )

    # convert-droid: convert DROID episodes into pretraining_v1 format
    droid_parser = subparsers.add_parser("convert-droid", help="Convert DROID episodes to pretraining format")
    droid_parser.add_argument("--source", default="cadene/droid_1.0.1", help="DROID repo ID")
    droid_parser.add_argument("--target", required=True, help="Target local repo ID, e.g. local/droid_2000_delta")
    droid_parser.add_argument("--n-episodes", type=int, default=2000, help="Number of episodes to convert (from ep 0)")
    droid_parser.add_argument("--resume", action="store_true", help="Resume an interrupted conversion")

    args = parser.parse_args()

    if args.command == "list":
        # When --match is used without an explicit --org, default to all MATCH_ORGS
        orgs = MATCH_ORGS if args.match is not None and args.org == ["LSY-lab"] else args.org
        list_hub_datasets(orgs, match_features_of=args.match, contains=args.contains)
    elif args.command == "features":
        show_features(args.repo_id)
    elif args.command == "compare":
        compare_dataset_features(args.datasets)
    elif args.command == "convert":
        convert_absolute_to_delta(args.source, args.target)
    elif args.command == "convert-droid":
        convert_droid_to_pretraining(args.source, args.target, episodes=list(range(args.n_episodes)), resume=args.resume)
    elif args.command == "merge":
        if args.sources is None:
            sources = MERGE_DATASETS_V2 if args.target.endswith("v2") else MERGE_DATASETS
        else:
            # When passed via CLI, task strings come from MERGE_DATASETS_V2 for known repos,
            # or default to an empty string for unknown ones.
            task_map = {repo_id: task for repo_id, task in MERGE_DATASETS_V2}
            sources = [(r, task_map.get(r, "")) for r in args.sources]
        merge_datasets(
            sources=sources,
            reference_repo_id=args.reference,
            target_repo_id=args.target,
            push=args.push,
            private=not args.public,
            resume=args.resume,
        )
