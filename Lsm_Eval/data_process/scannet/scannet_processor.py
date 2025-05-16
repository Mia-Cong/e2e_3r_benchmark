from data_process.base_processor import BaseSceneProcessor, BaseSceneProcessorConfig
import os
import numpy as np
import cv2
import json
import re
import torch
from typing import Dict, List
import argparse


class ScanNetConfig(BaseSceneProcessorConfig):
    def __init__(self, root_dir: str, save_dir: str, device: torch.device, num_workers: int = 16):
        super().__init__(root_dir, save_dir, device, num_workers)

class ScanNetProcessor(BaseSceneProcessor):
    def __init__(self, config: ScanNetConfig = None):
        super().__init__(config)
    
    def get_all_scene_paths(self) -> List[str]:
        return [os.path.join(self.config.root_dir, scene_name) for scene_name in os.listdir(self.config.root_dir)]
    
    def get_all_frame_paths(self, scene_path: str) -> Dict[str, List[str]]:
        folders = {
            'color': os.path.join(scene_path, "color"),
            'depth': os.path.join(scene_path, "depth"),
            'pose': os.path.join(scene_path, "pose")
        }
        
        # Get files and create frame number mappings
        frame_mappings = {
            key: {int(re.search(r'(\d+)\.', f).group(1)): f 
                for f in os.listdir(path) 
                if re.search(r'(\d+)\.', f)}
            for key, path in folders.items()
        }
        
        # Find common frames across all folders
        common_frames = sorted(set.intersection(*map(set, frame_mappings.values())))
        
        # Build result dictionary
        return {
            frame: [os.path.join(folders[k], frame_mappings[k][frame]) 
                    for k in folders]
            for frame in common_frames
        }
    
    def get_intrinsics(self, scene_path: str) -> dict:
        """
        Get intrinsics from scene_path
        """
        intrinsic_path = os.path.join(scene_path, "intrinsic", "intrinsic_depth.txt")
        intrinsic = np.loadtxt(intrinsic_path)
        intrinsic = torch.from_numpy(intrinsic).float().to(self.config.device)
        intrinsic = intrinsic[:3, :3]
        return intrinsic
    
    def get_instance_id_to_class_map(self, scene_path: str) -> dict:
        """
        Get instance id to class map from scene_path
        """
        aggregation_path = os.path.join(scene_path, f"{scene_path.split('/')[-1]}.aggregation.json")
        aggregation = json.load(open(aggregation_path))
        instance_id_to_class_map = {
            item['id'] + 1: item['label'] for item in aggregation['segGroups'] if item['label'] not in self.config.unused_classes}
        return instance_id_to_class_map
    
    def load_single_frame(self, frame_path: tuple) -> dict:
        """
        Load frame data from paths.
        input:
            frame_path: Tuple of (color_path, depth_path, pose_path)
        return:
            dict: Frame data including depth, color and pose data
        """
        try:
            color_path, depth_path, pose_path = frame_path
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            color_data = cv2.imread(color_path)
            pose_data = np.loadtxt(pose_path)
            
            # convert to torch tensors and move to device
            depth_data = torch.from_numpy(depth_data.astype(np.float32)).to(self.config.device)
            color_data = torch.from_numpy(color_data.astype(np.float32)).to(self.config.device)
            pose_data = torch.from_numpy(pose_data).float().to(self.config.device)
            
            return {
                'depth_data': depth_data,
                'color_data': color_data,
                'pose_data': pose_data
            }
        except Exception as e:
            # Ensure GPU memory is cleared even if an error occurs
            torch.cuda.empty_cache()
            raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="data/scannet_extracted")
    parser.add_argument("--save_dir", type=str, default="data/scannet_processed")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()
    config = ScanNetConfig(args.root_dir, args.save_dir, args.device, args.num_workers)
    processor = ScanNetProcessor(config)
    processor.process_all_scenes_parallel()
        