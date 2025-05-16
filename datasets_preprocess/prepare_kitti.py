# %%
#!/usr/bin/python

from PIL import Image
import numpy as np


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.
    return depth

# %%
import glob
import os
import shutil
depth_dirs = glob.glob("/ssd2/wenyan/3r_benchmark/monst3r/data/kitti/val/*/proj_depth/groundtruth/image_02")
for dir in depth_dirs:
    # new depth dir
    new_depth_dir = "/ssd2/wenyan/3r_benchmark/datasets/kitti/depth_selection/val_selection_cropped/groundtruth_depth_gathered/" + dir.split("/")[-4]+"_02"
    # print(new_depth_dir)
    new_image_dir = "/ssd2/wenyan/3r_benchmark/datasets/kitti/depth_selection/val_selection_cropped/image_gathered/" + dir.split("/")[-4]+"_02"
    os.makedirs(new_depth_dir, exist_ok=True)
    os.makedirs(new_image_dir, exist_ok=True)
    cnt=0
    for depth_file in sorted(glob.glob(dir + "/*.png"))[:110:2]: #../data/kitti/val/2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000005.png
        new_path = new_depth_dir + "/" + depth_file.split("/")[-1]
        shutil.copy(depth_file, new_path)
        # get the path of the corresponding image
        mid = "_".join(depth_file.split("/")[8].split("_")[:3])
        image_file = depth_file.replace('val', mid).replace('proj_depth/groundtruth/image_02', 'image_02/data')
        # print(image_file)
        # check if the image file exists
        if os.path.exists(image_file):
            new_path = new_image_dir + "/" + image_file.split("/")[-1]
            shutil.copy(image_file, new_path)
            cnt+=1
        else:
            print("Image file does not exist: ", image_file)
    print(dir, cnt)


