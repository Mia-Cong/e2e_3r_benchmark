import glob
import os
import shutil
import numpy as np
from tqdm import tqdm
from colmap_utils import read_model, qvec2rotmat, get_camera_matrix



def load_colmap_data(colmap_data_folder):
    # print('Loading COLMAP data...')
    input_format = '.bin'
    cameras, images, points3D = read_model(colmap_data_folder, ext=input_format)
    # print(f'num_cameras: {len(cameras)}')
    # print(f'num_images: {len(images)}')
    # print(f'num_points3D: {len(points3D)}')

    colmap_pose_dict = {}

    # Loop through COLMAP images
    for img_id, img_info in images.items():
        img_name = img_info.name

        C_R_G, C_t_G = qvec2rotmat(img_info.qvec), img_info.tvec

        # invert
        G_t_C = -C_R_G.T @ C_t_G
        G_R_C = C_R_G.T

        cam_info = cameras[img_info.camera_id]
        cam_params = cam_info.params
        K, _ = get_camera_matrix(camera_params=cam_params, camera_model=cam_info.model)

        colmap_pose_dict[img_name] = (K, G_R_C, G_t_C)

    return colmap_pose_dict, points3D



dirs = glob.glob("../data/ultrra/*/")
split_name = "ground_aerial"
for data_folder in dirs:
    input_image_folder = os.path.join(data_folder, 'images')
    save_image_folder = os.path.join(data_folder, 'image_pairs')
    colmap_folder = os.path.join(data_folder, 'model')
    pairs_file_txt = os.path.join(data_folder, f'{split_name}_covis_pairs.txt')
    pairs = []
    with open(pairs_file_txt, 'r') as f:
        for line in f:
            img1, img2 = line.strip().split()
            pairs.append((img1, img2))
    wriva_pose_dict, _ = load_colmap_data(colmap_folder)
    
    os.makedirs(save_image_folder, exist_ok=True)
    gt_poses = []
    for pair_idx in tqdm(range(len(pairs))):
        img1, img2 = pairs[pair_idx]
        img1_fullpath = os.path.join(input_image_folder, img1)
        img2_fullpath = os.path.join(input_image_folder, img2)

        if not os.path.exists(img1_fullpath) or not os.path.exists(img2_fullpath):
            print(f'Image not found: {img1_fullpath} or {img2_fullpath}')
            continue
        shutil.copy2(img1_fullpath, os.path.join(save_image_folder, "%03d_0.jpg" % pair_idx))
        shutil.copy2(img2_fullpath, os.path.join(save_image_folder, "%03d_1.jpg" % pair_idx))
        
        G_R_C1, G_t_C1 = wriva_pose_dict[img1][1], wriva_pose_dict[img1][2]
        G_R_C2, G_t_C2 = wriva_pose_dict[img2][1], wriva_pose_dict[img2][2]
        G_T_C1 = np.eye(4)
        G_T_C1[:3, :4] = np.hstack((G_R_C1, G_t_C1.reshape(3, 1)))
        G_T_C2 = np.eye(4)
        G_T_C2[:3, :4] = np.hstack((G_R_C2, G_t_C2.reshape(3, 1)))
        gt_poses.append(np.array([G_T_C1, G_T_C2], dtype=np.float32)) # 2x4x4
    np.save(os.path.join(data_folder, 'gt_poses.npy'), np.stack(gt_poses, axis=0))
            
        
        
    #assert False

    
    '''
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
    '''