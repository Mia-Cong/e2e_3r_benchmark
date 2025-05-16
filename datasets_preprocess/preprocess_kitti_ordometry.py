import os
import shutil
import numpy as np
import glob

# Define the original path to the KITTI dataset
dataset_path = "/ssd2/wenyan/3r_benchmark/datasets/kitti_ordometry_ori/dataset"

# Define the new destination path where the filtered images and poses will be saved
new_dataset_path = "/ssd2/wenyan/3r_benchmark/datasets/kitti_ordometry/"  # Update this path

# Process each sequence (00, 01, ..., 10)
for seq in range(11):
    # Get the image paths for left camera (image_2)
    img_paths = sorted(glob.glob(f"{dataset_path}/sequences/{seq:02d}/image_2/*.png"), key=lambda x: int(os.path.basename(x).split('.')[0]))
    
    # Define the path for the pose file
    pose_path = f"{dataset_path}/poses/{seq:02d}.txt"

    # Open and read the pose file
    with open(pose_path, 'r') as pose_file:
        poses = np.loadtxt(pose_file)
        
    # Select the corresponding poses (every other row from the first 110)
    selected_poses = poses[:110:2]

    # Create directories for the new images and poses in the new dataset path
    new_img_dir = os.path.join(new_dataset_path, f"sequences/{seq:02d}/image_55")
    os.makedirs(new_img_dir, exist_ok=True)

    # Select images (every other image from the first 110)
    selected_img_paths = img_paths[:110:2]

    # Copy the selected images to the new directory
    for i, img_path in enumerate(selected_img_paths):
        new_img_path = os.path.join(new_img_dir, f"frame_{i:04d}.png")
        shutil.copy(img_path, new_img_path)

    # Save the corresponding 55 poses
    new_pose_path = os.path.join(new_dataset_path, f"sequences/{seq:02d}/pose_55.txt")
    os.makedirs(os.path.dirname(new_pose_path), exist_ok=True)
    with open(new_pose_path, 'w') as f:
        for pose in selected_poses:
            f.write(f"{' '.join(map(str, pose))}\n")

    print(f"Sequence {seq:02d} processed. 55 images and poses saved to {new_dataset_path}.")


# import os
# import shutil
# import numpy as np
# import glob

# # Define the path to the KITTI dataset
# dataset_path = "/ssd2/wenyan/3r_benchmark/datasets/kitti_ordometry/dataset"

# # Process each sequence (00, 01, ..., 10)
# for seq in range(11):
#     # Get the image paths for left camera (image_2)
#     img_paths = sorted(glob.glob(f"{dataset_path}/sequences/{seq:02d}/image_2/*.png"), key=lambda x: int(os.path.basename(x).split('.')[0]))
    
#     # Define the path for the pose file
#     pose_path = f"{dataset_path}/poses/{seq:02d}.txt"

#     # Open and read the pose file
#     with open(pose_path, 'r') as pose_file:
#         poses = np.loadtxt(pose_file)
        
#     # Select the corresponding poses (every other row from the first 110)
#     selected_poses = poses[:110:2]
#     # Create directories for the new images and poses
#     new_img_dir = f"{dataset_path}/sequences/{seq:02d}/image_55"
#     os.makedirs(new_img_dir, exist_ok=True)

#     # Select images (every other image from the first 110)
#     selected_img_paths = img_paths[:110:2]

#     # Copy the selected images to the new directory
#     for i, img_path in enumerate(selected_img_paths):
#         new_img_path = os.path.join(new_img_dir, f"frame_{i:04d}.png")
#         shutil.copy(img_path, new_img_path)

#     # Save the corresponding 55 poses
#     new_pose_path = f"{dataset_path}/sequences/{seq:02d}/pose_55.txt"
#     with open(new_pose_path, 'w') as f:
#         for pose in selected_poses:
#             f.write(f"{' '.join(map(str, pose))}\n")

#     print(f"Sequence {seq:02d} processed. 55 images and poses saved.")
