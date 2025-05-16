import os
import shutil
import numpy as np
from PIL import Image
import pickle

def save_sample_data(sample, dest_root, source_root):
    """Save all images, depth maps, and poses to the destination folder."""
    sample_name = sample.name
    sample_base = sample.base

    # Copy all images
    if "images" in sample.data:
        for idx, image_obj in enumerate(sample.data["images"]):
            src_image_path = os.path.join(source_root, sample_base, image_obj.path)
            dest_image_dir = os.path.dirname(src_image_path).replace("RobustMVD", "RobustMVD_mvdsplit")
            os.makedirs(dest_image_dir, exist_ok=True)
            dest_image_path = os.path.join(dest_image_dir, os.path.basename(image_obj.path))
            print((src_image_path, dest_image_path))
            # breakpoint()
            # Check if the source image exists and then copy
            if os.path.exists(src_image_path):
                shutil.copy(src_image_path, dest_image_path)
            else:
                print(f"Warning: Source image {src_image_path} not found.")

    # Copy depth
    if "depth" in sample.data:
        depth_obj = sample.data["depth"]
        src_depth_path = os.path.join(source_root, sample_base, depth_obj.path)
        dest_depth_dir = os.path.dirname(src_depth_path).replace("RobustMVD", "RobustMVD_mvdsplit")
        os.makedirs(dest_depth_dir, exist_ok=True)
        dest_depth_path = os.path.join(dest_depth_dir, os.path.basename(depth_obj.path))
        print((src_depth_path, dest_depth_path))
        # Check if the source depth file exists and then copy
        if os.path.exists(src_depth_path):
            shutil.copy(src_depth_path, dest_depth_path)
        else:
            print(f"Warning: Source depth {src_depth_path} not found.")


def process_samples(pickle_file_path, dest_root, source_root):
    """Process each sample and save all images, depth maps, and poses to the destination while preserving the structure."""
    with open(pickle_file_path, "rb") as f:
        data = pickle.load(f)
    cnt=1
    for sample in data:
        print(f"Processing {cnt} sample {sample.name}...")
        
        save_sample_data(sample, dest_root, source_root)
        print(f"Sample {sample.name} saved to {dest_root}")
        # breakpoint()
        cnt+=1


# Example usage
pickle_file_path = "/ssd2/wenyan/3r_benchmark/robustmvd/rmvd/data/sample_lists/dtu.robustmvd.mvd.pickle"
dest_root = "/ssd2/wenyan/3r_benchmark/datasets/RobustMVD_mvdsplit/dtu"
source_root = "/ssd2/wenyan/3r_benchmark/datasets/RobustMVD/dtu"  # Root directory of the source dataset

process_samples(pickle_file_path, dest_root, source_root)
