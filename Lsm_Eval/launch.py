# --------------------------------------------------------
# training executable for DUSt3R
# --------------------------------------------------------

from large_spatial_model.utils.path_manager import init_all_submodules
init_all_submodules()

from large_spatial_model.model import LSM_Dust3R
from large_spatial_model.utils.visualization_utils import render_video_from_file

from dust3r.training import get_args_parser, load_model, train
from dust3r.pose_eval import eval_pose_estimation
# from dust3r.depth_eval import eval_mono_depth_estimation
import croco.utils.misc as misc  # noqa
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
from dust3r.model import AsymmetricCroCo3DStereo



if __name__ == '__main__':
    args = get_args_parser()

    args.add_argument('--file_list', type=str, nargs='+', required=True,
                        help='List of input image files or directories')
    args.add_argument('--model_path', type=str, required=True)
    args.add_argument('--output_path', type=str, required=True)
    args.add_argument('--resolution', type=int, default=256)
    args.add_argument('--n_interp', type=int, default=90)
    args.add_argument('--fps', type=int, default=30)


    args = args.parse_args()
    if args.mode.startswith('eval'):
        misc.init_distributed_mode(args)
        global_rank = misc.get_rank()
        world_size = misc.get_world_size()
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        model_path = args.pretrained

        # fix the seed
        seed = args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = args.cudnn_benchmark
        # model, _ = load_model(args, device)
        model = LSM_Dust3R.from_pretrained(model_path)
        model.eval()
        # model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
        os.makedirs(args.output_dir, exist_ok=True)

        if args.mode == 'eval_pose':
            ate_mean, rpe_trans_mean, rpe_rot_mean, outfile_list, bug = eval_pose_estimation(args, model, device, save_dir=args.output_dir)
            print(f'ATE mean: {ate_mean}, RPE trans mean: {rpe_trans_mean}, RPE rot mean: {rpe_rot_mean}')
        # if args.mode == 'eval_depth':
        #     eval_mono_depth_estimation(args, model, device)

        exit(0)
    # evaluation only
    # train(args)
