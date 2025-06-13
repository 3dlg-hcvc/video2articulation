import os
import glob
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
import cv2
import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.io import load_ply
from yourdfpy import URDF
import trimesh
import pickle
import argparse
import sys
from pathlib import Path
from collections import deque
from typing import Tuple, List, Dict

from utils import precompute_object2label, find_movable_part, precompute_camera2label


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


def descendant_links(robot: URDF, joint_name: str) -> list[str]:
    """Return the names of all links driven by *joint_name* (inclusive)."""
    if joint_name not in robot.joint_map:
        raise KeyError(f"joint '{joint_name}' not found")

    start_link = robot.joint_map[joint_name].child            # Link object
    q, visited = deque([start_link]), set()
    joint_list = robot.joint_map.values()

    while q:
        link = q.popleft()
        if link in visited:
            continue
        visited.add(link)

        # push every joint whose *parent* is this link
        for j in joint_list:
            if j.parent == link:
                q.append(j.child)

    return sorted(visited)


def load_joint_cfg(names_file: Path, values_file: Path) -> Dict[str, float]:
    """Return {joint_name: value} mapping from the two input files."""
    if not names_file.exists():
        sys.exit(f"Joint‑name file not found: {names_file}")
    if not values_file.exists():
        sys.exit(f"Joint‑value file not found: {values_file}")

    names: List[str] = [ln.strip() for ln in names_file.read_text().splitlines() if ln.strip()]
    vals = np.load(values_file)[0]

    if len(names) != len(vals):
        sys.exit(f"{len(names)} names vs {len(vals)} values – they must match.")

    return dict(zip(names, map(float, vals)))


def resolve_mesh_path(filename: str, urdf_dir: Path) -> Path:
    """Return absolute path, resolving relative paths against *urdf_dir*."""
    p = Path(filename)
    if not p.is_absolute():
        p = urdf_dir / p
    return p.resolve()


def sample_on_mesh(mesh: trimesh.Trimesh, n: int) -> np.ndarray:
    """Uniformly sample *n* points on *mesh* surface."""
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    return pts.astype(np.float32)


def process_visuals(
    robot: URDF,
    link_name: str,
    T_link: np.ndarray,
    urdf_dir: Path,
) -> List[trimesh.Trimesh]:
    """Load & sample each visual OBJ mesh of *link_name* given its world transform."""
    link = robot.link_map[link_name]
    mesh_list = []
    for idx, visual in enumerate(link.visuals):
        geom = visual.geometry.mesh
        if not hasattr(geom, "filename"):
            continue  # primitives

        mesh_path = resolve_mesh_path(geom.filename, urdf_dir)
        if not mesh_path.exists():
            print(f"[warn] mesh not found: {mesh_path}")
            continue

        mesh = trimesh.load_mesh(mesh_path, process=False)
        if getattr(geom, "scale", None):
            mesh.apply_scale(geom.scale)

        mesh.apply_transform(T_link @ visual.origin)
        mesh_list.append(mesh)
    return mesh_list


def sample_gt_pcd(link_name_list: list, robot: URDF, urdf_dir: str, final_pts_num: int = 10000) -> np.ndarray:
    print("Sampling point clouds …")
    all_meshes: List[trimesh.Trimesh] = []
    for link_name in link_name_list:  # type: ignore[attr-defined]
        print(f"Processing link: {link_name}")
        # yourdfpy returns transform from base frame (world) to `link` with cfg applied.
        T_link = robot.get_transform(link_name)  # frame_from defaults to base
        mesh_list = process_visuals(robot, link_name, T_link, urdf_dir)
        if len(mesh_list) > 0:
            all_meshes.extend(mesh_list)
    combined = trimesh.util.concatenate(all_meshes)              # a single TriangleMesh
    points, _ = trimesh.sample.sample_surface(combined, final_pts_num)

    rotate_back = R.from_euler('zyx', [90, 0, -90], degrees=True).as_matrix()
    merged = (rotate_back @ points.T).T
    return merged


def compute_geometry_error(obj_id: str, joint_id: int, joint_data_dir: str, results_dir: str, pred_joint_type: str, device: str = "cuda:0") -> Dict[str, float] | None:
    urdf_path = Path(f"partnet-mobility-v0/{obj_id}/mobility.urdf").expanduser().resolve()
    urdf_dir = urdf_path.parent

    print(f"Loading URDF from {urdf_path}")
    robot = URDF.load(urdf_path, mesh_dir=urdf_dir)

    joint_name = f"joint_{joint_id}"
    moving_links = descendant_links(robot, joint_name)
    if not moving_links:
        print(f"[warn] no children found for joint '{joint_name}'")
        return

    # Build and apply joint configuration.
    joint_name_list = f"{joint_data_dir}/joint_id_list.txt"
    joint_value_list = f"{joint_data_dir}/qpos.npy"
    cfg = load_joint_cfg(Path(joint_name_list), Path(joint_value_list))
    unknown = [j for j in cfg if j not in robot.joint_map]
    if unknown:
        sys.exit(f"Unknown joints in provided list: {', '.join(unknown)}")

    robot.update_cfg(cfg)  # ← sets internal configuration used by get_transform

    print("Sampling full point clouds …")
    link_map = robot.link_map
    full_link_list = link_map.keys()
    
    # chamfer distance on whole object
    gt_full_pcd = sample_gt_pcd(full_link_list, robot, urdf_dir, final_pts_num=10000)
    recon_full_pcd, _ = load_ply(f"{results_dir}/surface_pcd.ply")
    recon_full_pcd = recon_full_pcd.to(device)
    gt_full_pcd = torch.from_numpy(gt_full_pcd).to(device).to(recon_full_pcd.dtype)
    assert gt_full_pcd.shape[0] == gt_full_pcd.shape[0], "sample number not equal"
    bi_chamfer_dist = chamfer_distance(recon_full_pcd[None, ...], gt_full_pcd[None, ...])
    print(f"Chamfer Distance: {bi_chamfer_dist[0].item()}")

    # chamfer distance on moving part
    gt_moving_pcd = sample_gt_pcd(moving_links, robot, urdf_dir, final_pts_num=10000)
    gt_moving_pcd = torch.from_numpy(gt_moving_pcd).to(device).to(recon_full_pcd.dtype)
    if os.path.exists(f"{results_dir}/{pred_joint_type}/moving_pcd.ply"):
        recon_moving_pcd, _ = load_ply(f"{results_dir}/{pred_joint_type}/moving_pcd.ply")
        recon_moving_pcd = recon_moving_pcd.to(device)
        moving_chamfer_dist = chamfer_distance(recon_moving_pcd[None, ...], gt_moving_pcd[None, ...])
        print(f"Moving Chamfer Distance: {moving_chamfer_dist[0].item()}")
    else:
        print(f"[warn] no moving point cloud found for joint '{joint_name}'")
        moving_chamfer_dist = torch.tensor([1.0], device=device)

    # chamfer distance on static part
    static_links = [link for link in full_link_list if link not in moving_links]
    gt_static_pcd = sample_gt_pcd(static_links, robot, urdf_dir, final_pts_num=10000)
    gt_static_pcd = torch.from_numpy(gt_static_pcd).to(device).to(recon_full_pcd.dtype)
    if os.path.exists(f"{results_dir}/{pred_joint_type}/static_pcd.ply"):
        recon_static_pcd, _ = load_ply(f"{results_dir}/{pred_joint_type}/static_pcd.ply")
        recon_static_pcd = recon_static_pcd.to(device)
        static_chamfer_dist = chamfer_distance(recon_static_pcd[None, ...], gt_static_pcd[None, ...])
        print(f"Static Chamfer Distance: {static_chamfer_dist[0].item()}")
    else:
        print(f"[warn] no static point cloud found for joint '{joint_name}'")
        static_chamfer_dist = torch.tensor([1.0], device=device)

    geometry_error = {
        "full_chamfer_distance": bi_chamfer_dist[0].item(),
        "moving_chamfer_distance": moving_chamfer_dist[0].item(),
        "static_chamfer_distance": static_chamfer_dist[0].item()
    }
    return geometry_error


def main(args):
    meta_file = "new_partnet_mobility_dataset_correct_intr_meta.json"
    with open(meta_file, "r") as f:
        data_meta = json.load(f)
    joint_type_map = {"hinge": "revolute", "slider": "prismatic"}
    
    # init pc
    joint_data_dir = args.view_dir[:args.view_dir.rfind("view_")]
    actor_pose_path = f"{joint_data_dir}/actor_pose.pkl"
    with open(f"{joint_data_dir}/actor_pose.pkl", 'rb') as f:
        obj_pose_dict = pickle.load(f)
    actor_list = []
    for actor_name in obj_pose_dict.keys():
        actor_list.append(int(actor_name[6:]))
    
    surface_dir = f"{joint_data_dir}/view_init"
    surface_img = []
    surface_xyz = []
    surface_mask = []
    for i in range(24):
        img = cv2.imread("{}/rgb/{}".format(surface_dir, "%06d.png" % i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        surface_img.append(img)
        xyz = np.load("{}/xyz/{}".format(surface_dir, "%06d.npz" % i))['a']
        surface_xyz.append(xyz)
        segment = np.load("{}/segment/{}".format(surface_dir, "%06d.npz" % i))['a']
        mask = segment == -1
        for actor_id in actor_list:
            mask_id = segment == actor_id
            mask = np.logical_or(mask, mask_id)
        surface_mask.append(mask)
    surface_rgb = np.stack(surface_img).reshape(-1, 3)
    surface_xyz = np.stack(surface_xyz).reshape(-1, 3)
    surface_segment = np.stack(surface_mask).flatten()

    sample_num = min(480 * 64, surface_rgb.shape[0])
    sample_index = np.random.choice(np.arange(surface_rgb.shape[0]), sample_num, replace=False)
    surface_rgb = surface_rgb[sample_index]
    surface_xyz = surface_xyz[sample_index]
    surface_segment = surface_segment[sample_index]
    
    surface_xyz = surface_xyz[surface_segment]
    surface_rgb = surface_rgb[surface_segment]

    object2label = precompute_object2label(obj_pose_dict["actor_6"][0])
    surface_xyz = np.dot(surface_xyz, object2label[:3, :3].T) + object2label[:3, 3]

    # gt joint parameters
    opt_view = int(args.view_dir[args.view_dir.rfind("view_") + 5:args.view_dir.rfind("view_") + 6])
    cat = joint_data_dir.split('/')[2]
    obj_id = joint_data_dir.split('/')[3]
    interaction_list = data_meta[cat][obj_id]["interaction_list"]
    joint_id = int(joint_data_dir.split('/')[4][6:-3])
    interaction_dict = None
    for interaction in interaction_list:
        if joint_id == interaction["id"]:
            interaction_dict = interaction
    assert interaction_dict is not None, "does not find interaction"
    gt_joint_axis = interaction_dict["joint"]["axis"]["direction"]
    gt_joint_pos = interaction_dict["joint"]["axis"]["origin"]
    gt_joint_type = joint_type_map[interaction_dict["type"]]
    sample_rgb_index = [int(file_name[:-4]) for file_name in os.listdir(f"{args.view_dir}/sample_rgb/")]
    sample_rgb_index.sort()
    gt_joint_value = np.load(f"{joint_data_dir}/gt_joint_value.npy")
    sample_gt_joint_value = gt_joint_value[sample_rgb_index]

    # gt moving map
    sample_rgb_dir = f"{args.view_dir}/sample_rgb"
    segment_dir = f"{args.view_dir}/segment"
    img_list = os.listdir(sample_rgb_dir)
    img_list.sort()
    moving_part_id = find_movable_part(obj_pose_dict)
    gt_moving_map = []
    obj_mask_list = []
    for i in range(len(img_list)):
        segment = np.load(f"{segment_dir}/{img_list[i][:-4]}.npz")['a']
        dynamic_mask = segment == moving_part_id
        gt_moving_map.append(dynamic_mask)
        obj_mask = segment == -1
        for actor_id in actor_list:
            obj_mask_id = segment == actor_id
            obj_mask = np.logical_or(obj_mask, obj_mask_id)
        obj_mask_list.append(obj_mask)
    gt_moving_map_bool = np.stack(gt_moving_map) # N, H, W

    obj_mask_bool = np.stack(obj_mask_list) # N, H, W
    obj_mask = obj_mask_bool.astype(np.float64)
    gt_moving_map_bool = np.logical_and(gt_moving_map_bool, obj_mask_bool)
    gt_moving_map = gt_moving_map_bool.astype(np.float64)

    # camera se3 matrix
    gt_camera_pose = np.load(f"{args.view_dir}/camera_pose.npy")
    gt_camera_se3 = []
    for i in sample_rgb_index:
        gt_camera_se3_i= precompute_camera2label(gt_camera_pose[i], object2label)
        gt_camera_se3.append(gt_camera_se3_i)
    gt_camera_se3 = np.stack(gt_camera_se3) # numpy N, 4, 4

    # joint estimation
    if os.path.exists(f"{args.results_dir}/prismatic/best_loss.txt"):
        with open(f"{args.results_dir}/prismatic/best_loss.txt", 'r') as f:
            prismatic_loss = float(f.read().strip())
    else:
        prismatic_loss = 100
    if os.path.exists(f"{args.results_dir}/revolute/best_loss.txt"):
        with open(f"{args.results_dir}/revolute/best_loss.txt", 'r') as f:
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
        geometry_error
    else:
        revolute_joint_dir = f"{args.results_dir}/revolute"
        revolute_joint_pos = np.load(f"{revolute_joint_dir}/joint_pos.npy")
        revolute_joint_axis = np.load(f"{revolute_joint_dir}/joint_axis.npy")
        min_dist = distances_to_line(surface_xyz, revolute_joint_pos, revolute_joint_axis)
        if min_dist < 0.15:
            pred_joint_type = "revolute" if revolute_loss < prismatic_loss else "prismatic"
        else:
            pred_joint_type = "prismatic"
        
        pred_joint_axis = np.load(f"{args.results_dir}/{pred_joint_type}/joint_axis.npy")
        pred_joint_pos = np.load(f"{args.results_dir}/{pred_joint_type}/joint_pos.npy")
        pred_joint_value = np.load(f"{args.results_dir}/{pred_joint_type}/joint_value.npy")
        pred_camera_pose = np.load(f"{args.results_dir}/{pred_joint_type}/camera_poses.npy")
        pred_moving_map = np.load(f"{args.results_dir}/{pred_joint_type}/moving_map.npz")['a']
        pred_joint_parameter = (pred_joint_type, pred_joint_axis, pred_joint_pos, pred_joint_value)
        gt_joint_parameter = (gt_joint_type, np.array(gt_joint_axis), np.array(gt_joint_pos), sample_gt_joint_value)
        joint_error = compute_joint_error(gt_joint_parameter, pred_joint_parameter,
                                                      gt_camera_se3, pred_camera_pose, gt_moving_map, pred_moving_map)
        geometry_error = compute_geometry_error(obj_id, joint_id, joint_data_dir, args.results_dir, pred_joint_type, device=args.device)
        
    print(f"Joint estimation results: {joint_error}")
    print(f"Geometry error results: {geometry_error}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    main(args)
    