import argparse
import os
import subprocess
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, 
                       default=max(1, multiprocessing.cpu_count() // 2),
                       help='number of parallel workers')
    parser.add_argument('--force', action='store_true',
                       help='force reprocessing of already processed scenes')
    args = parser.parse_args()

    print(f"Preprocessing data from {args.input_dir} to {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    scans_dir = os.path.join(args.input_dir, 'scans')
    scene_dirs = [d for d in os.listdir(scans_dir) if os.path.isdir(os.path.join(scans_dir, d))]
    
    # Filter out already processed scenes
    if not args.force:
        scene_dirs = filter_processed_scenes(scene_dirs, args.output_dir)
        print(f"Found {len(scene_dirs)} scenes to process after filtering")
    
    # Create argument list for parallel processing
    process_args = [(scans_dir, scene_dir, args.output_dir, args.force) for scene_dir in scene_dirs]
    
    # Use process pool for parallel processing
    with Pool(processes=args.num_workers) as pool:
        list(tqdm(
            pool.imap(process_scene_wrapper, process_args),
            total=len(scene_dirs),
            desc="Preprocessing scenes"
        ))

def filter_processed_scenes(scene_dirs, output_dir):
    """Filter out already processed scenes"""
    unprocessed_scenes = []
    
    for scene_dir in scene_dirs:
        scene_output_dir = os.path.join(output_dir, scene_dir)
        
        if not is_scene_processed(scene_output_dir):
            unprocessed_scenes.append(scene_dir)
    
    return unprocessed_scenes

def is_scene_processed(scene_output_dir):
    """Check if a scene has been fully processed"""
    required_dirs = ['color', 'depth', 'pose']
    required_files = ['intrinsic/intrinsic_color.txt', 
                      'intrinsic/intrinsic_depth.txt', 
                      'intrinsic/extrinsic_color.txt', 
                      'intrinsic/extrinsic_depth.txt']
    
    # Check required directories
    for dir_name in required_dirs:
        dir_path = os.path.join(scene_output_dir, dir_name)
        if not os.path.isdir(dir_path):
            return False
        
        # Check if directory contains files
        if len(os.listdir(dir_path)) == 0:
            return False
    
    # Check required files
    for file_path in required_files:
        if not os.path.isfile(os.path.join(scene_output_dir, file_path)):
            return False
    
    # Check frame consistency across different types
    color_dir = os.path.join(scene_output_dir, 'color')
    depth_dir = os.path.join(scene_output_dir, 'depth')
    pose_dir = os.path.join(scene_output_dir, 'pose')
    
    color_count = len([f for f in os.listdir(color_dir) if f.endswith('.jpg')])
    depth_count = len([f for f in os.listdir(depth_dir) if f.endswith('.png')])
    pose_count = len([f for f in os.listdir(pose_dir) if f.endswith('.txt')])
    
    if color_count == 0 or depth_count == 0 or pose_count == 0:
        return False
    
    if color_count != depth_count or color_count != pose_count:
        return False
    
    return True

def process_scene_wrapper(args):
    """Wrapper function for handling multiple arguments"""
    return preprocess_scene(*args)

def preprocess_scene(scans_dir, scene_dir, output_dir, force=False):    
    # Create output directory for current scene
    scene_output_dir = os.path.join(output_dir, scene_dir)
    
    # Check if scene is already processed when not forcing reprocessing
    if not force and os.path.exists(scene_output_dir):
        if is_scene_processed(scene_output_dir):
            print(f"Scene {scene_dir} already processed, skipping...")
            return True
    
    os.makedirs(scene_output_dir, exist_ok=True)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    reader_path = os.path.join(current_dir, "reader.py")
    
    # Configure reader.py command with parameters
    # Note: reader.py needs to support these parameters
    cmd = [
        "python", reader_path,
        "--filename", os.path.join(scans_dir, scene_dir, scene_dir + '.sens'),
        "--output_path", scene_output_dir,
        "--export_depth_images",
        "--export_color_images",
        "--export_poses",
        "--export_intrinsics",
        "--use_parallel",
        "--frame_skip", "10",
        "--image_size", "480", "640"
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing scene {scene_dir}: {e}")
        return False

if __name__ == "__main__":
    main()