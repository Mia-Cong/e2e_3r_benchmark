import os
import os.path as osp
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
import numpy as np
import cv2
from dust3r.utils.image import imread_cv2
import itertools

class Scannetpp(BaseStereoViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert self.split == 'train' # just for training
        self.num_views = 3 # render third view
        self.max_images = 200
        self._load_data()
        
    def _load_data(self):
        # Traverse all the folders in the data_root
        scene_names = [folder for folder in os.listdir(self.ROOT) if os.path.isdir(os.path.join(self.ROOT, folder))]
        scene_names.sort()

        # merge all pairs and images
        pairs = [] # (scene_name, image_idx1, image_idx2)
        images = {} # scene_name -> list of image_paths
        for scene_name in scene_names:
            images_paths = os.listdir(osp.join(self.ROOT, scene_name, 'dslr', 'rgb_resized_undistorted'))
            if len(images_paths) > self.max_images:
                # uniformly sample images
                images_paths = images_paths[::len(images_paths) // self.max_images]
            images_paths.sort()
            frame_num = len(images_paths)
            scene_combinations = [(i, j)
                                for i, j in itertools.combinations(range(frame_num), 2)
                                if 0 < abs(i - j) <= 30 and abs(i - j) % 5 == 0]
            pairs.extend([(scene_name, *pair) for pair in scene_combinations])
            images[scene_name] = images_paths
            
        self.pairs = pairs
        self.images = images
        
    def __len__(self):
        return len(self.pairs)
    
    def _get_views(self, idx, resolution, rng):
        scene_name, image_idx1, image_idx2 = self.pairs[idx]
        image_idx1 = int(image_idx1)
        image_idx2 = int(image_idx2)
        image_idx3 = int((image_idx1 + image_idx2) / 2)
        views = []
        for view_idx in [image_idx1, image_idx2, image_idx3]:
            basename = os.path.basename(self.images[scene_name][view_idx]).split('.')[0]
            # Load RGB image
            rgb_path = osp.join(self.ROOT, scene_name, 'dslr', 'rgb_resized_undistorted', f'{basename}.JPG')
            rgb_image = imread_cv2(rgb_path)
            # Load depthmap
            depthmap_path = osp.join(self.ROOT, scene_name, 'dslr', 'render_depth', f'{basename}.png')
            depthmap = imread_cv2(depthmap_path, cv2.IMREAD_UNCHANGED)
            depthmap = depthmap.astype(np.float32) / 1000
            depthmap[~np.isfinite(depthmap)] = 0  # invalid
            # Load camera parameters
            meta_path = osp.join(self.ROOT, scene_name, 'dslr', 'camera', f'{basename}.npz')
            meta = np.load(meta_path)
            intrinsics = meta['intrinsic']
            camera_pose = meta['extrinsic']
            # crop if necessary
            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx)
            views.append(dict(
                img=rgb_image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=camera_pose.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                dataset='ScanNet++',
                label=scene_name + '_' + basename,
                instance=f'{str(idx)}_{str(view_idx)}',
            ))
        return views

if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = Scannetpp(split='train', ROOT="data/scannetpp_render", resolution=224, aug_crop=16)

    print(len(dataset))

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 3
        print(view_name(views[0]), view_name(views[1]), view_name(views[2]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1, 2]]
        cam_size = max(auto_cam_size(poses), 0.001)
        for view_idx in [0, 1, 2]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx*255, (1 - idx)*255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()