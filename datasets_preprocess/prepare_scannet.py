import glob
import os
import shutil
import numpy as np

seq_list = sorted(os.listdir("../data/scannetv2"))
dst_root = "/ssd2/wenyan/3r_benchmark/datasets/scannetv2"
for seq in seq_list:
    base_dir = f"../data/scannetv2/{seq}"
    img_pathes = sorted(glob.glob(f"../data/scannetv2/{seq}/color/*.jpg"), key=lambda x: int(os.path.basename(x).split('.')[0]))
    depth_pathes = sorted(glob.glob(f"../data/scannetv2/{seq}/depth/*.png"), key=lambda x: int(os.path.basename(x).split('.')[0]))
    pose_pathes = sorted(glob.glob(f"../data/scannetv2/{seq}/pose/*.txt"), key=lambda x: int(os.path.basename(x).split('.')[0]))
    print(f"{seq}: {len(img_pathes)} {len(depth_pathes)}")
    
    # Clean up old 90-sampled folders and pose file if they exist
    # for obsolete in ["color_90", "depth_90", "pose_90.txt"]:
    #     obsolete_path = os.path.join(base_dir, obsolete)
    #     if os.path.isdir(obsolete_path):
    #         print(f"Removing directory: {obsolete_path}")
    #         shutil.rmtree(obsolete_path)
    #     elif os.path.isfile(obsolete_path):
    #         print(f"Removing file: {obsolete_path}")
    #         os.remove(obsolete_path)

    new_color_dir = f"../data/scannetv2/{seq}/color_12k_stride30"
    new_depth_dir = f"../data/scannetv2/{seq}/depth_12k_stride30"

    new_img_pathes = img_pathes[:1200:30]
    new_depth_pathes = depth_pathes[:1200:30]
    new_pose_pathes = pose_pathes[:1200:30]
    print(f"{seq}: {len(new_img_pathes)} {len(new_depth_pathes)}")
    os.makedirs(new_color_dir, exist_ok=True)
    os.makedirs(new_depth_dir, exist_ok=True)

    for i, (img_path, depth_path) in enumerate(zip(new_img_pathes, new_depth_pathes)):
        shutil.copy(img_path, f"{new_color_dir}/frame_{i:04d}.jpg")
        shutil.copy(depth_path, f"{new_depth_dir}/frame_{i:04d}.png")

    pose_new_path = f"../data/scannetv2/{seq}/pose_12k_stride30.txt"
    with open(pose_new_path, 'w') as f:
        for i, pose_path in enumerate(new_pose_pathes):
            with open(pose_path, 'r') as pose_file:
                pose = np.loadtxt(pose_file)
                pose = pose.reshape(-1)
                f.write(f"{' '.join(map(str, pose))}\n")

    # ===== Copy to destination folder =====
    dst_scene_dir = os.path.join(dst_root, seq)
    os.makedirs(dst_scene_dir, exist_ok=True)

    shutil.copytree(new_color_dir, os.path.join(dst_scene_dir, "color_12k_stride30"), dirs_exist_ok=True)
    shutil.copytree(new_depth_dir, os.path.join(dst_scene_dir, "depth_12k_stride30"), dirs_exist_ok=True)
    shutil.copy(pose_new_path, os.path.join(dst_scene_dir, "pose_12k_stride30.txt"))