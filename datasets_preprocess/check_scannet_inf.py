import os
import numpy as np

root_dir = "/ssd2/wenyan/3r_benchmark/processed_datasets/scannetv2"  # change this to your actual path
# root_dir = "/ssd2/wenyan/3r_benchmark/processed_datasets/scannetv2_sub"
for scene in sorted(os.listdir(root_dir)):
    pose_path = os.path.join(root_dir, scene, "pose_12k_stride30.txt")
    if not os.path.exists(pose_path):
        continue

    try:
        poses = np.loadtxt(pose_path)
        if np.isinf(poses).any():
            # print(f"[⚠️ INF FOUND] {pose_path}")
            print(scene)
    except Exception as e:
        print(f"[❌ ERROR] {pose_path}: {e}")
