import argparse
import os, sys

from SensorData import SensorData, OptimizedSensorData

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--filename', required=True, help='path to sens file to read')
parser.add_argument('--output_path', required=True, help='path to output folder')
parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
parser.add_argument('--export_poses', dest='export_poses', action='store_true')
parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')
parser.add_argument('--use_parallel', dest='use_parallel', action='store_true', help='use parallel processing for faster export')
parser.add_argument('--frame_skip', type=int, default=1, help='process every nth frame (default: 1)')
parser.add_argument('--image_size', nargs=2, type=int, help='resize images to this size (height width)')
parser.set_defaults(export_depth_images=False, export_color_images=False, export_poses=False, export_intrinsics=False, use_parallel=False)

opt = parser.parse_args()
print(opt)


def main():
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)
    
    # load the data
    sys.stdout.write('loading %s...' % opt.filename)
    
    # Choose which class to use based on parallel processing option
    if opt.use_parallel:
        sd = OptimizedSensorData(opt.filename)
        export_depth = sd.export_depth_images_parallel
        export_color = sd.export_color_images_parallel
        export_poses = sd.export_poses_parallel
    else:
        sd = SensorData(opt.filename)
        export_depth = sd.export_depth_images
        export_color = sd.export_color_images
        export_poses = sd.export_poses
        
    sys.stdout.write('loaded!\n')
    
    # Convert image_size from (height, width) format if specified
    image_size = None
    if opt.image_size:
        image_size = tuple(opt.image_size)  # (height, width)
    
    if opt.export_depth_images:
        export_depth(os.path.join(opt.output_path, 'depth'), image_size=image_size, frame_skip=opt.frame_skip)
    if opt.export_color_images:
        export_color(os.path.join(opt.output_path, 'color'), image_size=image_size, frame_skip=opt.frame_skip)
    if opt.export_poses:
        export_poses(os.path.join(opt.output_path, 'pose'), frame_skip=opt.frame_skip)
    if opt.export_intrinsics:
        sd.export_intrinsics(os.path.join(opt.output_path, 'intrinsic'))


if __name__ == '__main__':
    main()