import os
import shutil
import numpy as np
import pickle

def create_dest_folder_structure(sample, dest_root):
    """Create folder structure in the destination directory."""
    sample_base_dir = sample.base  # base directory (e.g., "courtyard")
    sample_name = sample.name  # sample name (e.g., "key000001")
    
    # Construct the full destination path
    sample_dest_dir = os.path.join(dest_root, sample_base_dir)
    # Create the base directory
    os.makedirs(sample_dest_dir, exist_ok=True)

    # Create subdirectories based on the original structure
    os.makedirs(os.path.join(sample_dest_dir, 'ground_truth_depth', 'dslr_images'), exist_ok=True)
    os.makedirs(os.path.join(sample_dest_dir, 'images', 'dslr_images'), exist_ok=True)
    
    return sample_dest_dir

def save_sample_data(sample, dest_root, source_root):
    """Save all images, depth, and poses to the destination folder."""
    sample_base_dir = sample.base
    sample_name = sample.name

    # Create the destination directory structure
    sample_dest_dir = create_dest_folder_structure(sample, dest_root)
    # Copy all images
    if "images" in sample.data:
        for idx, image_obj in enumerate(sample.data["images"]):
            # Construct the full source path
            # breakpoint()
            src_image_path = os.path.join(source_root, sample_base_dir, image_obj.path)
            dest_image_path = os.path.join(sample_dest_dir, "images/dslr_images", os.path.basename(image_obj.path))
            print(src_image_path, dest_image_path)
            # Check if the source image exists and then copy
            if os.path.exists(src_image_path):
                shutil.copy(src_image_path, dest_image_path)
            else:
                print(f"Warning: Source image {src_image_path} not found.")

    # Copy depth
    if "depth" in sample.data:
        depth_obj = sample.data["depth"]
        # Construct the full source path
        src_depth_path = os.path.join(source_root, sample_base_dir, depth_obj.path)
        dest_depth_path = os.path.join(sample_dest_dir, "ground_truth_depth", "dslr_images", os.path.basename(depth_obj.path))

        # Check if the source depth file exists and then copy
        if os.path.exists(src_depth_path):
            shutil.copy(src_depth_path, dest_depth_path)
        else:
            print(f"Warning: Source depth {src_depth_path} not found.")

    # Copy poses
    # if "poses" in sample.data:
    #     for idx, pose in enumerate(sample.data["poses"]):
    #         dest_pose_path = os.path.join(sample_dest_dir, f"pose_{idx}.npy")
    #         np.save(dest_pose_path, pose)

def process_samples(pickle_file_path, dest_root, source_root):
    """Process each sample and save all images, depth, and poses to the destination."""
    with open(pickle_file_path, "rb") as f:
        data = pickle.load(f)

    for sample in data:
        print(f"Processing sample {sample.name}...")
        save_sample_data(sample, dest_root, source_root)
        print(f"Sample {sample.name} saved to {dest_root}")

# Example usage
pickle_file_path = "/ssd2/wenyan/3r_benchmark/robustmvd/rmvd/data/sample_lists/eth3d.robustmvd.mvd.pickle"
dest_root = "/ssd2/wenyan/3r_benchmark/datasets/RobustMVD_mvdsplit/eth3d"
source_root = "/ssd2/wenyan/3r_benchmark/datasets/RobustMVD/eth3d"  # Root directory of the source dataset

process_samples(pickle_file_path, dest_root, source_root)
