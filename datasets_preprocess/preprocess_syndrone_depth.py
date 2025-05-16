import os
import json
import shutil
from tqdm import tqdm
from PIL import Image
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def generate_invalid_depth_mask(png_path, output_mask_path=None):
    # Load the 16-bit depth image
    depth_img = Image.open(png_path)
    depth_np = np.array(depth_img)

    # Create a mask where depth is 65535
    invalid_mask = (depth_np == 65535)
    valid_mask = 1-invalid_mask

    if output_mask_path:
        # Save the mask as an 8-bit PNG (0 = valid, 255 = invalid)
        mask_to_save = (valid_mask.astype(np.uint8)) * 255
        Image.fromarray(mask_to_save).save(output_mask_path)


def euler_to_quaternion(pitch, yaw, roll):
    """
    Converts pitch, yaw, roll (in degrees) to a quaternion (qx, qy, qz, qw)
    Note: pitch is rotation around x, yaw around y, roll around z
    """
    r = R.from_euler('xyz', [math.radians(pitch), math.radians(yaw), math.radians(roll)])
    return r.as_quat()  # returns in (x, y, z, w)

def sample_images_and_poses(root_dir, dest_root, num_samples=50, stride=2):
    scenes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    for scene in tqdm(scenes, desc="Processing scenes"):
        rgb_dir = os.path.join(root_dir, scene, "ClearNoon", "height20m", "rgb")
        depth_dir = os.path.join(root_dir, scene, "ClearNoon", "height20m", "depth")
        cam_dir = os.path.join(root_dir, scene, "ClearNoon", "height20m", "camera")

        if not os.path.exists(rgb_dir) or not os.path.exists(cam_dir):
            print(f"[Skip] Missing rgb or camera folder in {scene}")
            continue

        # Collect image file names (without extension)
        image_ids = sorted([f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
        selected_ids = image_ids[:num_samples * stride:stride]

        scene_dest_rgb = os.path.join(dest_root, scene, "rgb_50")
        scene_dest_depth = os.path.join(dest_root, scene, "depth_50")
        scene_dest_mask = os.path.join(dest_root, scene, "mask_50")
        scene_dest_pose = os.path.join(dest_root, scene, "pose_50.txt")
        os.makedirs(scene_dest_rgb, exist_ok=True)
        os.makedirs(scene_dest_depth, exist_ok=True)
        os.makedirs(scene_dest_mask, exist_ok=True)

        pose_lines = []

        for img_id in selected_ids:
            src_img_path = os.path.join(rgb_dir, img_id + ".jpg")
            src_depth_path = os.path.join(depth_dir, img_id + ".png")
            src_cam_path = os.path.join(cam_dir, img_id + ".json")

            if not os.path.exists(src_img_path) or not os.path.exists(src_depth_path) or not os.path.exists(src_cam_path):
                print(f"[Warning] Missing file for {img_id} in {scene}")
                continue

            shutil.copy(src_img_path, os.path.join(scene_dest_rgb, img_id + ".jpg"))
            shutil.copy(src_depth_path, os.path.join(scene_dest_depth, img_id + ".png"))
            generate_invalid_depth_mask(src_depth_path, os.path.join(scene_dest_mask, img_id + ".png"))

            with open(src_cam_path, 'r') as f:
                cam_data = json.load(f)

            x, y, z = cam_data["x"], cam_data["y"], cam_data["z"]
            pitch, yaw, roll = cam_data["pitch"], cam_data["yaw"], cam_data["roll"]
            qx, qy, qz, qw = euler_to_quaternion(pitch, yaw, roll)

            pose_lines.append(f"{img_id} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")

        # Save poses for this scene
        with open(scene_dest_pose, 'w') as f:
            f.write("\n".join(pose_lines))

# Example usage
sample_images_and_poses(
    root_dir="/ssd2/wenyan/3r_benchmark/datasets/syndrone_ori/renders",
    dest_root="/ssd2/wenyan/3r_benchmark/datasets/syndrone/",
    num_samples=50,
    stride=2
)
