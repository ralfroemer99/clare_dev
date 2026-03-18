import sys
# sys.path.insert(0, "lerobot_lsy/src")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from gym_libero.libero.utils.download_utils import libero_dataset_download
# for suite in ['libero_object', 'libero_goal']:
#     libero_dataset_download(datasets=suite)

# ds = LeRobotDataset(
#     repo_id="cadene/droid_1.0.1",
#     episodes=list(range(2000)),
#     video_backend="pyav",
# )
# print(f"Downloaded {ds.num_episodes} episodes, {len(ds)} frames.")

# for i in range(10):
#     repo_id = f"continuallearning/libero_10_image_task_{i}"
#     print(f"Downloading {repo_id}...")
#     ds = LeRobotDataset(repo_id=repo_id)
#     print(f"  -> {ds.num_episodes} episodes, {len(ds)} frames.")
ds = LeRobotDataset(repo_id="continuallearning/libero_spatial_image_task_0")
ds.push_to_hub(push_videos=False, private=False)
