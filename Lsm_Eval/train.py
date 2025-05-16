from large_spatial_model.utils.path_manager import init_all_submodules
init_all_submodules()

# replace inference.loss_of_one_batch
import dust3r.inference
from large_spatial_model.loss import loss_of_one_batch
dust3r.inference.loss_of_one_batch = loss_of_one_batch

# replace losses.Regr3D
import dust3r.losses
from large_spatial_model.loss import KWRegr3D
dust3r.losses.Regr3D = KWRegr3D

# replace losses.GaussianLoss
from large_spatial_model.loss import GaussianLoss
dust3r.losses.GaussianLoss = GaussianLoss

from dust3r.training import get_args_parser as dust3r_get_args_parser  # noqa
from dust3r.training import train  # noqa

import dust3r.datasets
from large_spatial_model.datasets.scannet import Scannet
from large_spatial_model.datasets.scannetpp import Scannetpp
dust3r.datasets.Scannetpp = Scannetpp
dust3r.datasets.Scannet = Scannet

from large_spatial_model.model import LSM_Dust3R
dust3r.training.LSM_Dust3R = LSM_Dust3R

import yaml

def get_args_parser():
    parser = dust3r_get_args_parser()
    parser.prog = 'LSM_Dust3R training'
    
    # Load the configuration
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Convert the config dict to a string of keyword arguments
    config_str = "config=" + str(config)
    
    # Set the default model string with parameters
    parser.set_defaults(model=f"LSM_Dust3R({config_str})")
    
    return parser


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    train(args)