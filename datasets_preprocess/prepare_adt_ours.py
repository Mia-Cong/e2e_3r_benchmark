import os
import os.path as osp
import shutil
import json
import glob
from tqdm import tqdm
import numpy as np
from loguru import logger as guru

def prepare_adt_dataset(source_dir, target_dir, sampling_rate=2):
    """
    Preprocess the ADT dataset by downsampling
    
    Args:
        source_dir: Source data directory, e.g. 'data/adt/Apartment_release_meal_seq133_5'
        target_dir: Target data directory, e.g. 'data/adt/Apartment_release_meal_seq133_5/prepare_all_s2'
        sampling_rate: Sampling rate, default is 2 frames per second
    """
    
    # Ensure the source directory exists
    if not osp.exists(source_dir):
        guru.error(f"Source directory does not exist: {source_dir}")
        return False
        
    # Read all frames data
    rgb_dir = osp.join(source_dir, "rgb")
    camera_dir = osp.join(source_dir, "camera")
    
    # Get all frame names
    frame_names = sorted([osp.splitext(f)[0] for f in os.listdir(rgb_dir) if f.endswith('.png')])
    
    # Control the maximum number of frames
    max_frames = 10000
    total_frames = len(frame_names)
    indices = list(range(0, total_frames, sampling_rate))
    if len(indices) > max_frames:
        indices = indices[:max_frames]

    # Create necessary subdirectories
    for subdir in ['rgb', 'camera']:
        os.makedirs(osp.join(target_dir, subdir), exist_ok=True)
        
    # Copy sampled data
    for idx in tqdm(indices, desc="Copy sampled frames"):
        frame_name = frame_names[idx]
        
        # Copy RGB image
        src_img = osp.join(rgb_dir, f"{frame_name}.png")
        dst_img = osp.join(target_dir, f"rgb/{frame_name}.png")
        shutil.copy(src_img, dst_img)
        
        # Copy camera parameters
        src_camera = osp.join(camera_dir, f"{frame_name}.json")
        dst_camera = osp.join(target_dir, f"camera/{frame_name}.json")
        if osp.exists(src_camera):
            shutil.copy(src_camera, dst_camera)

    # Generate preview video
    def create_video(input_dir, output_path, fps=10):
        try:
            import subprocess
            
            # Check if the input directory exists
            if not osp.exists(input_dir):
                guru.warning(f"Input directory does not exist: {input_dir}")
                return False
                
            # Build ffmpeg command
            cmd = f'/usr/bin/ffmpeg -y -framerate {fps} -pattern_type glob -i "{input_dir}/*.png" ' \
                  f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ' \
                  f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p ' \
                  f'-movflags +faststart -b:v 5000k "{output_path}"'
            
            # Execute command
            guru.info(f"Running command: {cmd}")
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if result.returncode != 0:
                guru.error(f"FFmpeg command failed: {result.stderr.decode('utf-8')}")
                return False
                
            guru.info(f"Video saved to: {output_path}")
            return True
        except Exception as e:
            guru.error(f"Error generating video: {str(e)}")
            return False
    
    # Generate RGB preview video
    create_video(
        osp.join(target_dir, "rgb"),
        osp.join(target_dir, "rgb.mp4")
    )
    
    guru.info(f"Dataset preprocessing completed, saved to: {target_dir}")
    return True

def process_all_adt_datasets(root_dir, sampling_rate=2):
    """
    Process all ADT datasets in the root directory
    
    Args:
        root_dir: Root directory of ADT dataset
        sampling_rate: Sampling rate, default is 2 frames per second
    """
    # Ensure the root directory exists
    if not osp.exists(root_dir):
        guru.error(f"Root directory does not exist: {root_dir}")
        return False
    
    # Get all possible dataset directories
    datasets = [d for d in os.listdir(root_dir) 
               if osp.isdir(osp.join(root_dir, d)) and 
               'release' in d]
    
    if not datasets:
        guru.warning(f"No datasets found in {root_dir}")
        return False
    
    # Process each dataset
    for dataset in datasets:
        source_dir = osp.join(root_dir, dataset)
        target_dir = osp.join(source_dir, f"prepare_all_stride{sampling_rate}")
        
        guru.info(f"Processing dataset: {dataset}")
        prepare_adt_dataset(source_dir, target_dir, sampling_rate)
    
    guru.info("All datasets processed")
    return True

if __name__ == "__main__":
    root_dir = "/ssd2/wenyan/3r_benchmark/Easi3R/data/adt/"
    process_all_adt_datasets(root_dir, sampling_rate=10)