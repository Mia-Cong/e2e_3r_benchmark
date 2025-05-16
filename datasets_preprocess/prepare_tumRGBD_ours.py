import glob
import os
import shutil
import numpy as np
import cv2

def read_file_list(filename):
    """
    Reads a trajectory from a text file. 
    
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
    
    Input:
    filename -- File name
    
    Output:
    dict -- dictionary of (stamp,data) tuples
    
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

def associate(first_list, second_list, offset, max_difference):
    """
    Associate two dictionaries of (stamp, data). As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.
    
    Input:
    first_list -- first dictionary of (stamp, data) tuples
    second_list -- second dictionary of (stamp, data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation

    Output:
    matches -- list of matched tuples ((stamp1, data1), (stamp2, data2))
    """
    # Convert keys to sets for efficient removal
    first_keys = set(first_list.keys())
    second_keys = set(second_list.keys())
    
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()
    return matches

dirs = glob.glob("/ssd2/wenyan/3r_benchmark/datasets/tum_rgbd/*/")
dirs = sorted(dirs)

STRIDE = 30

# extract frames
for dir in dirs:
    frames = []
    depth_frames = []
    gt = []
    first_file = dir + 'rgb.txt'
    second_file = dir + 'groundtruth.txt'
    depth_file = dir + 'depth.txt'

    first_list = read_file_list(first_file)
    second_list = read_file_list(second_file)
    depth_list = read_file_list(depth_file)
    
    matches = associate(first_list, second_list, 0.0, 0.02)
    
    final_matches = []
    for rgb_time, gt_time in matches:
        depth_times = np.array(list(depth_list.keys()))
        closest_idx = np.argmin(np.abs(depth_times - rgb_time))
        depth_time = depth_times[closest_idx]
        final_matches.append((rgb_time, gt_time, depth_time))
    
    frames = []
    depth_frames = []
    gt = []
    for a, b, d in final_matches:
        frames.append(dir + first_list[a][0])
        depth_frames.append(dir + depth_list[d][0])
        gt.append([b] + second_list[b])

    frames_sampled = frames[:1200:STRIDE]
    depth_frames_sampled = depth_frames[:1200:STRIDE]
    gt_sampled = gt[:1200:STRIDE]
    
    new_rgb_dir = dir + f'rgb_12k_stride{STRIDE}/'
    new_depth_dir = dir + f'depth_12k_stride{STRIDE}/'
    
    if os.path.exists(new_rgb_dir):
        shutil.rmtree(new_rgb_dir)
    if os.path.exists(new_depth_dir):
        shutil.rmtree(new_depth_dir)
        
    os.makedirs(new_rgb_dir, exist_ok=True)
    os.makedirs(new_depth_dir, exist_ok=True)

    for frame, depth_frame in zip(frames_sampled, depth_frames_sampled):
        try:
            shutil.copy(frame, new_rgb_dir)
            shutil.copy(depth_frame, new_depth_dir)
        except Exception as e:
            print(f"Error copying files: {str(e)}")
            continue

    groundtruth_file = dir + f'groundtruth_12k_stride{STRIDE}.txt'
    with open(groundtruth_file, 'w') as f:
        for pose in gt_sampled:
            line = f"{' '.join(map(str, pose))}\n"
            f.write(line)

    video_name = dir + f'rgb_12k_stride{STRIDE}.mp4'
    depth_video_name = dir + f'depth_12k_stride{STRIDE}.mp4'
    
    # Use ffmpeg to generate RGB video
    frame_files = sorted(glob.glob(new_rgb_dir + '*.png'))
    if frame_files:
        try:
            import subprocess
            
            # Build ffmpeg command - RGB video
            cmd = f'/usr/bin/ffmpeg -y -framerate 10 -pattern_type glob -i "{new_rgb_dir}*.png" ' \
                  f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ' \
                  f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p ' \
                  f'-movflags +faststart -b:v 5000k "{video_name}"'
            
            # Execute command
            print(f"Running command: {cmd}")
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if result.returncode != 0:
                print(f"FFmpeg command failed: {result.stderr.decode('utf-8')}")
            else:
                print(f"RGB video saved to: {video_name}")
        except Exception as e:
            print(f"Error generating RGB video: {str(e)}")

    # Use ffmpeg to generate depth video
    depth_files = sorted(glob.glob(new_depth_dir + '*.png'))
    if depth_files:
        try:
            import subprocess
            
            # Build ffmpeg command - depth video
            cmd = f'/usr/bin/ffmpeg -y -framerate 10 -pattern_type glob -i "{new_depth_dir}*.png" ' \
                  f'-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ' \
                  f'-vcodec h264 -preset fast -profile:v baseline -pix_fmt yuv420p ' \
                  f'-movflags +faststart -b:v 5000k "{depth_video_name}"'
            
            # Execute command
            print(f"Running command: {cmd}")
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            if result.returncode != 0:
                print(f"FFmpeg command failed: {result.stderr.decode('utf-8')}")
            else:
                print(f"Depth video saved to: {depth_video_name}")
        except Exception as e:
            print(f"Error generating depth video: {str(e)}")