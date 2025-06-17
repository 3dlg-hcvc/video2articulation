import os
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.io import load_ply
import argparse
from typing import Tuple, Dict

from data import SimDataLoader


def distances_to_line(points, line_point, line_dir, return_min=False):
    """
    Compute perpendicular distances from a set of 3‑D points to a 3‑D line.

    Parameters
    ----------
    points : (N, 3) array_like
        Coordinates of the N points.
    line_point : (3,) array_like
        A point on the line (the vector **a** in the formula).
    line_dir : (3,) array_like
        Direction vector of the line (the vector **v**).  Need not be unit‑length.
    return_min : bool, default False
        If True, also return the index of the closest point and its distance.

    Returns
    -------
    dists : (N,) ndarray
        Perpendicular distances for every point.
    (optional) min_idx, min_dist : int, float
        Index of the closest point and its distance (only if return_min=True).
    """
    p = np.asarray(points, dtype=float)          # (N, 3)
    a = np.asarray(line_point, dtype=float)      # (3,)
    v = np.asarray(line_dir, dtype=float)        # (3,)

    # Vector from line point to every point
    r = p - a                                    # (N, 3)

    # Norm of cross‑product gives numerator; norm of v gives denominator
    cross = np.cross(r, v)                       # (N, 3)
    num = np.linalg.norm(cross, axis=1)          # (N,)
    denom = np.linalg.norm(v)                    # scalar
    dists = num / denom

    min_dist = np.min(dists)
    if return_min:
        min_idx = np.argmin(dists)
        return dists, min_idx, dists[min_idx]
    return min_dist


def compute_joint_error(gt_joint_parameter: Tuple[str, np.ndarray, np.ndarray, np.ndarray], pred_joint_parameter: Tuple[str, np.ndarray, np.ndarray, np.ndarray], 
                             gt_camera_se3: np.ndarray, pred_camera_pose: np.ndarray, gt_moving_map: np.ndarray, pred_moving_map: np.ndarray) -> Dict[str, float | bool]:
    gt_joint_type = gt_joint_parameter[0]
    gt_joint_axis = gt_joint_parameter[1]
    gt_joint_pos = gt_joint_parameter[2]
    gt_joint_value = gt_joint_parameter[3]
    # joint estimation
    pred_joint_type = pred_joint_parameter[0]
    pred_joint_axis = pred_joint_parameter[1]
    pred_joint_axis = pred_joint_axis / np.linalg.norm(pred_joint_axis)
    pred_joint_pos = pred_joint_parameter[2]
    pred_joint_value = pred_joint_parameter[3]

    joint_ori_error = np.arccos(np.abs(np.dot(pred_joint_axis, gt_joint_axis)))

    n = np.cross(pred_joint_axis, gt_joint_axis)
    joint_pos_error = np.abs(np.dot(n, (pred_joint_pos - gt_joint_pos))) / np.linalg.norm(n)

    if gt_joint_type == "prismatic":
        joint_pos_error = 0
    
    joint_state_error = np.mean(np.abs(np.abs(gt_joint_value) - np.abs(pred_joint_value)))

    # camera estimation
    pred_camera_rotation = R.from_quat(pred_camera_pose[:, :4], scalar_first=True).as_matrix()
    pred_camera_translation = pred_camera_pose[:, 4:]
    rotation_error_matrix = pred_camera_rotation @ gt_camera_se3[:, :3, :3].transpose(0, 2, 1)
    cam_rotation_error = np.mean(np.arccos((np.trace(rotation_error_matrix, axis1=1, axis2=2) - 1) / 2))
    cam_translation_error = np.mean(np.linalg.norm(pred_camera_translation - gt_camera_se3[:, :3, 3], axis=1))

    # moving map
    intersection = gt_moving_map * pred_moving_map
    union = gt_moving_map + pred_moving_map - intersection
    valid_union_index = np.nonzero(np.sum(union, axis=(1, 2)) > 1e-3)[0] # remove empty union
    intersection = intersection[valid_union_index, :, :]
    union = union[valid_union_index, :, :]
    soft_iou = np.mean(np.sum(intersection, axis=(1, 2)) / np.sum(union, axis=(1, 2)))

    evaluation_results = {
        "joint orientation error": float(joint_ori_error),
        "joint position error": float(joint_pos_error),
        "joint state error": float(joint_state_error),
        "joint type error": pred_joint_type != gt_joint_type,
        "camera position error": float(cam_translation_error),
        "camera rotation error": float(cam_rotation_error),
        "moving map mIOU": float(soft_iou)
    }

    return evaluation_results


def compute_geometry_error(gt_full_pcd: np.ndarray, gt_moving_pcd: np.ndarray, gt_static_pcd: np.ndarray, results_dir: str, pred_joint_type: str, device: str = "cuda:0") -> Dict[str, float]:
    recon_full_pcd, _ = load_ply(f"{results_dir}/surface_pcd.ply")
    recon_full_pcd = recon_full_pcd.to(device)
    gt_full_pcd = torch.from_numpy(gt_full_pcd).to(device).to(recon_full_pcd.dtype)
    assert gt_full_pcd.shape[0] == gt_full_pcd.shape[0], "sample number not equal"
    bi_chamfer_dist = chamfer_distance(recon_full_pcd[None, ...], gt_full_pcd[None, ...])

    # chamfer distance on moving part
    gt_moving_pcd = torch.from_numpy(gt_moving_pcd).to(device).to(recon_full_pcd.dtype)
    if os.path.exists(f"{results_dir}/{pred_joint_type}/moving_pcd.ply"):
        recon_moving_pcd, _ = load_ply(f"{results_dir}/{pred_joint_type}/moving_pcd.ply")
        recon_moving_pcd = recon_moving_pcd.to(device)
        moving_chamfer_dist = chamfer_distance(recon_moving_pcd[None, ...], gt_moving_pcd[None, ...])
    else:
        print(f"[warn] no moving point cloud found")
        moving_chamfer_dist = torch.tensor([1.0], device=device)

    # chamfer distance on static part
    gt_static_pcd = torch.from_numpy(gt_static_pcd).to(device).to(recon_full_pcd.dtype)
    if os.path.exists(f"{results_dir}/{pred_joint_type}/static_pcd.ply"):
        recon_static_pcd, _ = load_ply(f"{results_dir}/{pred_joint_type}/static_pcd.ply")
        recon_static_pcd = recon_static_pcd.to(device)
        static_chamfer_dist = chamfer_distance(recon_static_pcd[None, ...], gt_static_pcd[None, ...])
    else:
        print(f"[warn] no static point cloud found")
        static_chamfer_dist = torch.tensor([1.0], device=device)

    geometry_error = {
        "full_chamfer_distance": bi_chamfer_dist[0].item(),
        "moving_chamfer_distance": moving_chamfer_dist[0].item(),
        "static_chamfer_distance": static_chamfer_dist[0].item()
    }
    return geometry_error


def main(args):
    data_loader = SimDataLoader(args.view_dir, None, args.meta_file_path, args.partnet_mobility_dir)
    surface_rgb, surface_xyz = data_loader.load_obj_surface(sample_num=480 * 64, return_segments=False)

    gt_joint_type, gt_joint_axis, gt_joint_pos, gt_joint_value = data_loader.load_gt_joint_params()

    gt_moving_map_list = data_loader.load_gt_moving_map()
    gt_moving_map = np.stack(gt_moving_map_list).astype(np.float64)  # N, H, W

    gt_camera_se3 = data_loader.load_gt_camera_pose_se3()

    gt_full_pcd, gt_moving_pcd, gt_static_pcd = data_loader.load_gt_pcd()

    # joint estimation
    if os.path.exists(f"{args.refinement_results_dir}/prismatic/best_loss.txt"):
        with open(f"{args.refinement_results_dir}/prismatic/best_loss.txt", 'r') as f:
            prismatic_loss = float(f.read().strip())
    else:
        prismatic_loss = 100
    if os.path.exists(f"{args.refinement_results_dir}/revolute/best_loss.txt"):
        with open(f"{args.refinement_results_dir}/revolute/best_loss.txt", 'r') as f:
            revolute_loss = float(f.read().strip())
    else:
        revolute_loss = 100
    
    if prismatic_loss == 100 and revolute_loss == 100:
        print("No valid joint refinement results found.")
        joint_error = {
            "joint orientation error": np.pi / 2,
            "joint position error": 1,
            "joint state error": 1 if gt_joint_type == "prismatic" else np.pi / 2,
            "joint type error": True,
            "camera position error": 1,
            "camera rotation error": np.pi / 2,
            "moving map mIOU": 0
        }
        geometry_error = {
            "full_chamfer_distance": 1,
            "moving_chamfer_distance": 1,
            "static_chamfer_distance": 1
        }
    else:
        revolute_joint_dir = f"{args.refinement_results_dir}/revolute"
        revolute_joint_pos = np.load(f"{revolute_joint_dir}/joint_pos.npy")
        revolute_joint_axis = np.load(f"{revolute_joint_dir}/joint_axis.npy")
        min_dist = distances_to_line(surface_xyz, revolute_joint_pos, revolute_joint_axis)
        if min_dist < 0.15:
            pred_joint_type = "revolute" if revolute_loss < prismatic_loss else "prismatic"
        else:
            pred_joint_type = "prismatic"
        
        pred_joint_axis = np.load(f"{args.refinement_results_dir}/{pred_joint_type}/joint_axis.npy")
        pred_joint_pos = np.load(f"{args.refinement_results_dir}/{pred_joint_type}/joint_pos.npy")
        pred_joint_value = np.load(f"{args.refinement_results_dir}/{pred_joint_type}/joint_value.npy")
        pred_camera_pose = np.load(f"{args.refinement_results_dir}/{pred_joint_type}/camera_poses.npy")
        pred_moving_map = np.load(f"{args.refinement_results_dir}/{pred_joint_type}/moving_map.npz")['a']
        pred_joint_parameter = (pred_joint_type, pred_joint_axis, pred_joint_pos, pred_joint_value)
        gt_joint_parameter = (gt_joint_type, np.array(gt_joint_axis), np.array(gt_joint_pos), gt_joint_value)
        joint_error = compute_joint_error(gt_joint_parameter, pred_joint_parameter,
                                                      gt_camera_se3, pred_camera_pose, gt_moving_map, pred_moving_map)
        geometry_error = compute_geometry_error(gt_full_pcd, gt_moving_pcd, gt_static_pcd, args.refinement_results_dir, pred_joint_type, device=args.device)
        
    print(f"Joint estimation results: {joint_error}")
    print(f"Geometry error results: {geometry_error}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view_dir", type=str, required=True)
    parser.add_argument("--meta_file_path", type=str, default="new_partnet_mobility_dataset_correct_intr_meta.json")
    parser.add_argument("--partnet_mobility_dir", type=str, default="partnet-mobility-v0")
    parser.add_argument("--refinement_results_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    main(args)
    