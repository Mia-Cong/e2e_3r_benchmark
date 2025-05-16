import os
import shutil

src_root = "/ssd2/wenyan/3r_benchmark/monst3r/data/tum"
dst_root = "/ssd2/wenyan/3r_benchmark/datasets/tum"

os.makedirs(dst_root, exist_ok=True)

seq_list = sorted(os.listdir(src_root))
for seq in seq_list:
    src_seq_path = os.path.join(src_root, seq)
    dst_seq_path = os.path.join(dst_root, seq)

    # Skip non-directories
    if not os.path.isdir(src_seq_path):
        continue

    os.makedirs(dst_seq_path, exist_ok=True)

    for subitem in ["depth_all_stride30", "rgb_all_stride30"]:
        src_subdir = os.path.join(src_seq_path, subitem)
        dst_subdir = os.path.join(dst_seq_path, subitem)
        if os.path.isdir(src_subdir):
            shutil.copytree(src_subdir, dst_subdir, dirs_exist_ok=True)
            print(f"Copied folder: {src_subdir} -> {dst_subdir}")

    # Copy the txt file
    txt_file = os.path.join(src_seq_path, "groundtruth_all_stride30.txt")
    if os.path.isfile(txt_file):
        shutil.copy(txt_file, os.path.join(dst_seq_path, "groundtruth_all_stride30.txt"))
        print(f"Copied file: {txt_file} -> {dst_seq_path}/groundtruth_all_stride30.txt")
