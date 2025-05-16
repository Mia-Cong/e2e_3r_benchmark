import glob
import os
import shutil
import numpy as np
from tqdm import tqdm

cls_dirs = glob.glob("/mnt/vita-nas/3rbenchmark/co3d/*/")
cls_dirs = sorted(cls_dirs)
# for catetogory:
for cls_dir in tqdm(cls_dirs):
    dirs = glob.glob(os.path.join(cls_dir, "*"))
    #print(cls_dir, dirs) 
    for dir in dirs:
        gt_path = os.path.join(dir, "gt_10.txt")
        # dir would be something like '../data/co3d/apple/598_92033_183486'
        frames = glob.glob(os.path.join(dir, "images_10", "*.jpg"))
        cameras = glob.glob(os.path.join(dir, "images_10", "*.npz"))
        frames = sorted(frames)
        cameras = sorted(cameras)
        assert len(frames) == len(cameras)
        # load camera params
        camera_poses = None
        for camera in cameras:
            input_metadata = np.load(camera)
            camera_pose = input_metadata['camera_pose'].astype(np.float32)
            camera_pose = camera_pose[:3, :4].flatten()
            #assert False, camera_pose.shape
            if camera_poses is None:
                camera_poses = camera_pose[None]
            else:
                camera_poses = np.concatenate([camera_poses, camera_pose[None]], axis=0)
        np.savetxt(gt_path, camera_poses)
        #assert False