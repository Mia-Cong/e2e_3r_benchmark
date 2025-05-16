import os
import glob
import random
import shutil
import numpy as np
from PIL import Image

re10k_video_root = "/ssd2/wenyan/3r_benchmark/datasets/rel10k_ori/test"
re10k_txt_root = "/ssd2/wenyan/3r_benchmark/datasets/rel10k_ori/metadata"
output_root = "/ssd2/wenyan/3r_benchmark/datasets/rel10k"  # Change this to your desired output folder
n_samples = 10  # Number of frames to sample per folder


# All video folders
all_folders = sorted(os.listdir(re10k_video_root))
all_folders = [f for f in all_folders if os.path.isdir(os.path.join(re10k_video_root, f))]

# If subset_file is provided, filter the folders
# Process each sequence folder
for seq in all_folders:
    folder_path = os.path.join(re10k_video_root, seq)
    txt_path = os.path.join(re10k_txt_root, seq + ".txt")

    if not os.path.exists(txt_path):
        continue

    with open(txt_path, "r") as f:
        txt_lines = f.read().strip().split("\n")
    if len(txt_lines) <= 1:
        continue
    txt_lines = txt_lines[1:]  # Skip first line (URL)

    # Mapping frame_id to pose info
    lines_map = {}
    for line in txt_lines:
        parts = line.strip().split()
        if len(parts) < 19:
            continue
        frame_id = parts[0]
        lines_map[frame_id] = parts

    frame_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    if len(frame_files) < 2:
        continue

    # Sample up to n_samples frames per folder
    n_to_sample = min(n_samples, len(frame_files))
    sampled_frames = random.sample(frame_files, n_to_sample)
    sampled_frames.sort()

    # Prepare paths for saving the sampled images and poses
    new_img_dir = os.path.join(output_root, f"{seq}/image_10")
    os.makedirs(new_img_dir, exist_ok=True)
    new_pose_path = os.path.join(output_root, f"{seq}/pose_10.txt")
    os.makedirs(os.path.dirname(new_pose_path), exist_ok=True)

    # Store selected views (images + poses)
    with open(new_pose_path, 'w') as f_out:
        for i, frame_path in enumerate(sampled_frames):
            basename = os.path.splitext(os.path.basename(frame_path))[0]
            if basename not in lines_map:
                continue

            columns = lines_map[basename]
            pose_str = " ".join(columns)
            # print(pose_str)
            f_out.write(f"{pose_str}\n")

            # Load image and save
            shutil.copy(frame_path, new_img_dir)
            # img_rgb = Image.open(frame_path)
            # if img_rgb is None:
            #     continue

            # # Save image
            # new_img_path = os.path.join(new_img_dir, f"frame_{i:04d}.png")
            # img_rgb.save(new_img_path)

        print(f"Sequence {seq} processed: {n_to_sample} frames and poses saved.")