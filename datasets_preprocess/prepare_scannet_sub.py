import os
import shutil
import random

src_root = "/ssd2/wenyan/3r_benchmark/processed_datasets/scannetv2"  # Update with your actual path
dst_root = "/ssd2/wenyan/3r_benchmark/processed_datasets/scannetv2_sub"  # Destination folder

os.makedirs(dst_root, exist_ok=True)

# List all scene folders
all_scenes = [d for d in os.listdir(src_root) if d.startswith("scene") and os.path.isdir(os.path.join(src_root, d))]

# Randomly select 25 scenes
selected_scenes = random.sample(all_scenes, 25)

# Copy selected scenes
for scene in selected_scenes:
    src = os.path.join(src_root, scene)
    dst = os.path.join(dst_root, scene)
    shutil.copytree(src, dst)
    print(f"Copied {scene}")

print("Done.")
