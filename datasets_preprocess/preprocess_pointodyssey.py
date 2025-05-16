import numpy as np
import os
import os.path as osp
from glob import glob
from tqdm import tqdm 
import shutil

blender2opencv = np.float32([[1, 0, 0, 0],
                             [0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]])

input_path = "/ssd2/wenyan/3r_benchmark/datasets/PointOdyssey"
output_path = "/ssd2/wenyan/3r_benchmark/processed_datasets/PointOdyssey"
set_list = ["val"]
for set in set_list:
    data_dir = os.path.join(input_path, set)
    out_dir = os.path.join(output_path, set)
    os.makedirs(out_dir, exist_ok=True)

    for sequence in tqdm(sorted(os.listdir(data_dir))):
      if len(sequence.split('.'))==1:
        print(sequence)
        seq_savepath = osp.join(out_dir, sequence)
        os.makedirs(seq_savepath, exist_ok=True)

        imgs_path = osp.join(data_dir, sequence, "rgbs")
        depths_path = osp.join(data_dir, sequence, "depths")
        annotations = np.load(osp.join(data_dir, sequence, "anno.npz"))
        trajs_3d = annotations['trajs_3d'].astype(np.float32)
        intrinsics = annotations['intrinsics'].astype(np.float32)
        extrinsics = annotations['extrinsics'].astype(np.float32)

        rgbfiles_sel = sorted(os.listdir(imgs_path))[:110:2]
        depthfiles_sel = sorted(os.listdir(depths_path))[:110:2]
        extrinsics_sel = extrinsics[:110:2]


        # Create directories for the new images and poses in the new dataset path
        new_img_dir = os.path.join(seq_savepath, f"rgbs_55")
        os.makedirs(new_img_dir, exist_ok=True)
        new_depth_dir = os.path.join(seq_savepath, f"depths_55")
        os.makedirs(new_depth_dir, exist_ok=True)
        # Save the corresponding 55 poses
        new_pose_path = os.path.join(seq_savepath, f"pose_55.txt")
        os.makedirs(os.path.dirname(new_pose_path), exist_ok=True)
        f = open(new_pose_path, 'w')

        for rgbfile, depthfile, i in zip(rgbfiles_sel, depthfiles_sel, range(len(extrinsics_sel))):
            extrinsic_matrix = extrinsics_sel[i].reshape(4, 4).astype(np.float32) @ blender2opencv
            f.write(f"{' '.join(map(str, extrinsic_matrix.flatten()))}\n")

            new_img_path = os.path.join(new_img_dir, f"{i*2:05d}.jpg")
            shutil.copy(osp.join(imgs_path, rgbfile), new_img_path)

            new_depth_path = os.path.join(new_depth_dir, f"{i*2:05d}.jpg")
            shutil.copy(osp.join(depths_path, depthfile), new_depth_path)
