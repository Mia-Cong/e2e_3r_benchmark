import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import time
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from torch.utils.data import DataLoader
# from add_ckpt_path import add_path_to_dust3r
from accelerate import Accelerator
from torch.utils.data._utils.collate import default_collate
import tempfile
from tqdm import tqdm

from large_spatial_model.utils.path_manager import init_all_submodules
init_all_submodules()

from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo, inf
from dust3r.training import get_args_parser

from large_spatial_model.model import LSM_Dust3R

def load_model(args, device):
    print('Loading model: {:s}'.format(args.model))
    model = eval(args.model)
    model.to(device)
    model_without_ddp = model
    if args.weights and not args.resume:
        print('Loading pretrained: ', args.weights)
        ckpt = torch.load(args.weights, map_location=device)
        print(model.load_state_dict(ckpt['model'], strict=False))
        del ckpt  # in case it occupies memory

    return model, model_without_ddp

def my_get_args_parser():
    print('Getting args parser')
    # parser = argparse.ArgumentParser("3D Reconstruction evaluation", add_help=False)
    parser = get_args_parser()
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="ckpt name",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument(
        "--conf_thresh", type=float, default=0.0, help="confidence threshold"
    )
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--revisit", type=int, default=1, help="revisit times")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--dense", action="store_true")

    return parser

def wrap_scene_into_preds(scene, batch, device):
    from dust3r.utils.geometry import geotrf, inv
    preds = []
    pts3d = scene.get_pts3d()
    first_pose = scene.get_im_poses()[0]
    for pts in pts3d:
        pts_at_first_frame = geotrf(inv(first_pose), pts)
        preds.append(
            {'pts3d_in_other_view': pts_at_first_frame.unsqueeze(0).to(device)}
        )
    for i, view in enumerate(batch):
        for k, v in view.items():
            if isinstance(v, torch.Tensor):
                batch[i][k] = view[k].to(device)
    return preds, batch


def main(args):
    # add_path_to_dust3r(args.weights)
    from eval.mv_recon.data import SevenScenes, NRGBD, DTU, TUM, SCANNET
    from eval.mv_recon.utils import accuracy, completion

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    else:
        raise NotImplementedError
    datasets_all = {
        "7scenes": SevenScenes(
            split="test",
            ROOT="/data/3r/7scenes",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=200 if not args.dense else 20,
        ), 
        "NRGBD": NRGBD(
            split="test",
            ROOT="/data/3r/neural_rgbd_data",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=500 if not args.dense else 40,
        ),
        'DTU': DTU(
            split='test', 
            ROOT="/data/3r/dtu_test_mvsnet_release",
            resolution=resolution, 
            num_seq=1, 
            full_video=True, 
            kf_every=16 if not args.dense else 5),
        'TUM': TUM(
            split="test",
            ROOT="/data/3r/tum_rgbd",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=10 if not args.dense else 1),
        'SCANNET': SCANNET(
            split="test",
            ROOT="/data/3r/scannetv2_sub",
            resolution=resolution,
            num_seq=1,
            full_video=True,
            kf_every=10 if not args.dense else 1),
    }

    accelerator = Accelerator()
    device = accelerator.device
    model_name = args.model_name
    if model_name == "ours" or model_name == "cut3r":
        from dust3r.model import ARCroco3DStereo
        from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
        from dust3r.utils.geometry import geotrf
        from copy import deepcopy

        model = ARCroco3DStereo.from_pretrained(args.weights).to(device)
        model.eval()
    elif model_name == "dust3r" or model_name == "mast3r" or model_name == "monst3r":
        from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
        from dust3r.utils.geometry import geotrf
        from dust3r.model import AsymmetricCroCo3DStereo, inf
        from copy import deepcopy

        # model, _ = load_model(args, device)
        # load lsm model
        model = LSM_Dust3R.from_pretrained(args.weights)
        model.eval()
    else:
        raise NotImplementedError
    os.makedirs(args.output_dir, exist_ok=True)

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)

    for name_data, dataset in datasets_all.items():
        if args.eval_dataset != name_data:
            continue
        # output dir + dataset name
        save_path = osp.join(args.output_dir, name_data)
        os.makedirs(save_path, exist_ok=True)
        log_file = osp.join(save_path, f"logs_{accelerator.process_index}.txt")

        acc_all = 0
        acc_all_med = 0
        comp_all = 0
        comp_all_med = 0
        nc1_all = 0
        nc1_all_med = 0
        nc2_all = 0
        nc2_all_med = 0

        fps_all = []
        time_all = []

        with accelerator.split_between_processes(list(range(len(dataset)))) as idxs:
            for data_idx in tqdm(idxs):
                # ['img', 'depthmap', 'camera_pose', 'camera_intrinsics', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'pts3d', 'valid_mask', 'img_mask', 'ray_mask', 'ray_map', 'update', 'reset', 'rng']
                batch = default_collate([dataset[data_idx]])
                ignore_keys = set(
                    [
                        "depthmap",
                        "dataset",
                        "label",
                        "instance",
                        "idx",
                        "true_shape",
                        "rng",
                    ]
                )
                scene_id = batch[0]['label'][0].rsplit("/", 1)[0]
                metric_path = os.path.join(save_path, f"{scene_id.replace('/', '_')}_eval_metrics.txt")
                if os.path.exists(metric_path):
                    print(f'Metric file {metric_path} already exists, skipping sequence {scene_id}.')
                    continue
                for view in batch:
                    for name in view.keys():  # pseudo_focal
                        if name in ignore_keys:
                            continue
                        if isinstance(view[name], tuple) or isinstance(
                            view[name], list
                        ):
                            view[name] = [
                                x.to(device, non_blocking=True) for x in view[name]
                            ]
                        else:
                            view[name] = view[name].to(device, non_blocking=True)
                # for cut3r
                if model_name == "ours" or model_name == "cut3r":
                    revisit = args.revisit
                    update = not args.freeze
                    if revisit > 1:
                        # repeat input for 'revisit' times
                        new_views = []
                        for r in range(revisit):
                            for i in range(len(batch)):
                                new_view = deepcopy(batch[i])
                                new_view["idx"] = [
                                    (r * len(batch) + i)
                                    for _ in range(len(batch[i]["idx"]))
                                ]
                                new_view["instance"] = [
                                    str(r * len(batch) + i)
                                    for _ in range(len(batch[i]["instance"]))
                                ]
                                if r > 0:
                                    if not update:
                                        new_view["update"] = torch.zeros_like(
                                            batch[i]["update"]
                                        ).bool()
                                new_views.append(new_view)
                        batch = new_views
                    with torch.cuda.amp.autocast(enabled=False):
                        start = time.time()
                        output = model(batch)
                        end = time.time()
                        preds, batch = output.ress, output.views
                    valid_length = len(preds) // revisit
                    preds = preds[-valid_length:]
                    batch = batch[-valid_length:]
                    fps = len(batch) / (end - start)
                    print(
                        f"Finished reconstruction for {name_data} {data_idx+1}/{len(dataset)}, FPS: {fps:.2f}"
                    )
                    # continue
                    fps_all.append(fps)
                    time_all.append(end - start)

                    # Evaluation
                    print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")
                    gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                        criterion.get_all_pts3d_t(batch, preds)
                    )
                    pred_scale, gt_scale, pred_shift_z, gt_shift_z = (
                        monitoring["pred_scale"],
                        monitoring["gt_scale"],
                        monitoring["pred_shift_z"],
                        monitoring["gt_shift_z"],
                    )

                    in_camera1 = None
                    pts_all = []
                    pts_gt_all = []
                    images_all = []
                    masks_all = []
                    conf_all = []

                    for j, view in enumerate(batch):
                        if in_camera1 is None:
                            in_camera1 = view["camera_pose"][0].cpu()

                        image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                        mask = view["valid_mask"].cpu().numpy()[0]

                        # pts = preds[j]['pts3d' if j==0 else 'pts3d_in_other_view'].detach().cpu().numpy()[0]
                        pts = pred_pts[j].cpu().numpy()[0]
                        conf = preds[j]["conf"].cpu().data.numpy()[0]
                        # mask = mask & (conf > 1.8)

                        pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                        H, W = image.shape[:2]
                        cx = W // 2
                        cy = H // 2
                        l, t = cx - 112, cy - 112
                        r, b = cx + 112, cy + 112
                        image = image[t:b, l:r]
                        mask = mask[t:b, l:r]
                        pts = pts[t:b, l:r]
                        pts_gt = pts_gt[t:b, l:r]

                        #### Align predicted 3D points to the ground truth
                        pts[..., -1] += gt_shift_z.cpu().numpy().item()
                        pts = geotrf(in_camera1, pts)

                        pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()
                        pts_gt = geotrf(in_camera1, pts_gt)

                        images_all.append((image[None, ...] + 1.0) / 2.0)
                        pts_all.append(pts[None, ...])
                        pts_gt_all.append(pts_gt[None, ...])
                        masks_all.append(mask[None, ...])
                        conf_all.append(conf[None, ...])
                
                elif model_name == "dust3r" or model_name == "mast3r" or model_name == "monst3r":
                    silent = True
                    # for dust3r like model: dust3r, mast3r, monst3r
                    revisit = args.revisit
                    update = not args.freeze
                    if revisit > 1:
                        # repeat input for 'revisit' times
                        new_views = []
                        for r in range(revisit):
                            for i in range(len(batch)):
                                new_view = deepcopy(batch[i])
                                new_view["idx"] = [
                                    (r * len(batch) + i)
                                    for _ in range(len(batch[i]["idx"]))
                                ]
                                new_view["instance"] = [
                                    str(r * len(batch) + i)
                                    for _ in range(len(batch[i]["instance"]))
                                ]
                                if r > 0:
                                    if not update:
                                        new_view["update"] = torch.zeros_like(
                                            batch[i]["update"]
                                        ).bool()
                                new_views.append(new_view)
                        batch = new_views

                    if model_name == "dust3r" or model_name == "mast3r":
                        print(f"==> loading dust3r model with scene graph complete")
                        pairs = make_pairs(batch, scene_graph='complete', prefilter=None, symmetrize=True)
                    elif model_name == "monst3r":
                        scene_graph_type = args.scene_graph_type
                        print(f"==> loading dust3r model with scene graph {scene_graph_type}")
                        pairs = make_pairs(batch, scene_graph=scene_graph_type, prefilter=None, symmetrize=True)

                    output = inference(pairs, model, device, batch_size=1, verbose=not silent)

                    mode = GlobalAlignerMode.PointCloudOptimizer if len(batch) > 2 else GlobalAlignerMode.PairViewer

                    scene = global_aligner(output, device=device, mode=mode)

                    lr = 0.01
                    loss = scene.compute_global_alignment(init='mst', niter=300, schedule='linear', lr=lr)

                    preds, batch = wrap_scene_into_preds(scene, batch, device)

                    # Evaluation
                    print(f"Evaluation for {name_data} {data_idx+1}/{len(dataset)}")
                    gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
                        criterion.get_all_pts3d_t(batch, preds)
                    )
                    pred_scale, gt_scale, pred_shift_z, gt_shift_z = (
                        monitoring["pred_scale"],
                        monitoring["gt_scale"],
                        monitoring["pred_shift_z"],
                        monitoring["gt_shift_z"],
                    )

                    in_camera1 = None
                    pts_all = []
                    pts_gt_all = []
                    images_all = []
                    masks_all = []
                    conf_all = []

                    for j, view in enumerate(batch):
                        if in_camera1 is None:
                            in_camera1 = view["camera_pose"][0].cpu()

                        image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                        mask = view["valid_mask"].cpu().numpy()[0]

                        # pts = preds[j]['pts3d' if j==0 else 'pts3d_in_other_view'].detach().cpu().numpy()[0]
                        pts = pred_pts[j].detach().cpu().numpy()[0]
                        # conf = preds[j]["conf"].cpu().data.numpy()[0]
                        # mask = mask & (conf > 1.8)

                        pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                        H, W = image.shape[:2]
                        cx = W // 2
                        cy = H // 2
                        l, t = cx - 112, cy - 112
                        r, b = cx + 112, cy + 112
                        image = image[t:b, l:r]
                        mask = mask[t:b, l:r]
                        pts = pts[t:b, l:r]
                        pts_gt = pts_gt[t:b, l:r]

                        #### Align predicted 3D points to the ground truth
                        pts[..., -1] += gt_shift_z.cpu().numpy().item()
                        pts = geotrf(in_camera1, pts)

                        pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()
                        pts_gt = geotrf(in_camera1, pts_gt)

                        images_all.append((image[None, ...] + 1.0) / 2.0)
                        pts_all.append(pts[None, ...])
                        pts_gt_all.append(pts_gt[None, ...])
                        masks_all.append(mask[None, ...])
                        # conf_all.append(conf[None, ...])

                images_all = np.concatenate(images_all, axis=0)
                pts_all = np.concatenate(pts_all, axis=0)
                pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                masks_all = np.concatenate(masks_all, axis=0)

                scene_id = view["label"][0].rsplit("/", 1)[0]

                save_params = {}

                save_params["images_all"] = images_all
                save_params["pts_all"] = pts_all
                save_params["pts_gt_all"] = pts_gt_all
                save_params["masks_all"] = masks_all

                np.save(
                    os.path.join(save_path, f"{scene_id.replace('/', '_')}.npy"),
                    save_params,
                )

                if "DTU" in name_data:
                    threshold = 100
                else:
                    threshold = 0.1

                pts_all_masked = pts_all[masks_all > 0]
                pts_gt_all_masked = pts_gt_all[masks_all > 0]
                images_all_masked = images_all[masks_all > 0]

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(
                    pts_all_masked.reshape(-1, 3)
                )
                pcd.colors = o3d.utility.Vector3dVector(
                    images_all_masked.reshape(-1, 3)
                )
                o3d.io.write_point_cloud(
                    os.path.join(
                        save_path, f"{scene_id.replace('/', '_')}-mask.ply"
                    ),
                    pcd,
                )

                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(
                    pts_gt_all_masked.reshape(-1, 3)
                )
                pcd_gt.colors = o3d.utility.Vector3dVector(
                    images_all_masked.reshape(-1, 3)
                )
                o3d.io.write_point_cloud(
                    os.path.join(save_path, f"{scene_id.replace('/', '_')}-gt.ply"),
                    pcd_gt,
                )

                trans_init = np.eye(4)

                reg_p2p = o3d.pipelines.registration.registration_icp(
                    pcd,
                    pcd_gt,
                    threshold,
                    trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                )

                transformation = reg_p2p.transformation

                pcd = pcd.transform(transformation)
                pcd.estimate_normals()
                pcd_gt.estimate_normals()

                gt_normal = np.asarray(pcd_gt.normals)
                pred_normal = np.asarray(pcd.normals)

                acc, acc_med, nc1, nc1_med = accuracy(
                    pcd_gt.points, pcd.points, gt_normal, pred_normal
                )
                comp, comp_med, nc2, nc2_med = completion(
                    pcd_gt.points, pcd.points, gt_normal, pred_normal
                )
                print(
                    f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                )
                print(
                    f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}",
                    file=open(log_file, "a"),
                )
                with open(metric_path, 'w') as f:
                    f.write(f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}")

                acc_all += acc
                comp_all += comp
                nc1_all += nc1
                nc2_all += nc2

                acc_all_med += acc_med
                comp_all_med += comp_med
                nc1_all_med += nc1_med
                nc2_all_med += nc2_med

                # release cuda memory
                torch.cuda.empty_cache()

                # break

            # accelerator.wait_for_everyone()
            # Get depth from pcd and run TSDFusion
            if accelerator.is_main_process:
                to_write = ""
                # Copy the error log from each process to the main error log
                # for i in range(8):
                #     if not os.path.exists(osp.join(save_path, f"logs_{i}.txt")):
                #         break
                #     with open(osp.join(save_path, f"logs_{i}.txt"), "r") as f_sub:
                #         to_write += f_sub.read()

                to_write = ""
                # read from _eval_metrics.txt
                for file in os.listdir(save_path):
                    if file.endswith('_eval_metrics.txt'):
                        with open(osp.join(save_path, file), "r") as f:
                            to_write += f.read()
                            to_write += "\n"

                with open(osp.join(save_path, f"logs_all.txt"), "w") as f:
                    log_data = to_write
                    metrics = defaultdict(list)
                    for line in log_data.strip().split("\n"):
                        match = regex.match(line)
                        if match:
                            data = match.groupdict()
                            # Exclude 'scene_id' from metrics as it's an identifier
                            for key, value in data.items():
                                if key != "scene_id":
                                    metrics[key].append(float(value))
                            metrics["nc"].append(
                                (float(data["nc1"]) + float(data["nc2"])) / 2
                            )
                            metrics["nc_med"].append(
                                (float(data["nc1_med"]) + float(data["nc2_med"])) / 2
                            )
                    mean_metrics = {
                        metric: sum(values) / len(values)
                        for metric, values in metrics.items()
                    }

                    c_name = "mean"
                    print_str = f"{c_name.ljust(20)}: "
                    for m_name in mean_metrics:
                        print_num = np.mean(mean_metrics[m_name])
                        print_str = print_str + f"{m_name}: {print_num:.5f} | "
                    print_str = print_str + "\n"
                    f.write(to_write + print_str)


from collections import defaultdict
import re

pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":

    parser = my_get_args_parser()
    args = parser.parse_args()

    main(args)
