import numpy as np
import cv2
import trimesh
from scipy.spatial.transform import Rotation as R
from yourdfpy import URDF
import sys
from pathlib import Path
from collections import deque

from typing import Dict, List


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


def resolve_urdf_mesh_path(filename: str, urdf_dir: Path) -> Path:
    """Return absolute path, resolving relative paths against *urdf_dir*."""
    p = Path(filename)
    if not p.is_absolute():
        p = urdf_dir / p
    return p.resolve()


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


def process_visuals(
    urdf_dir: Path,
    robot: URDF,
    link_name: str,
    T_link: np.ndarray,
) -> List[trimesh.Trimesh]:
    """Load each visual OBJ mesh of *link_name* given its world transform."""
    link = robot.link_map[link_name]
    mesh_list = []
    for idx, visual in enumerate(link.visuals):
        geom = visual.geometry.mesh
        if not hasattr(geom, "filename"):
            continue  # primitives

        mesh_path = resolve_urdf_mesh_path(geom.filename, urdf_dir)
        if not mesh_path.exists():
            print(f"[warn] mesh not found: {mesh_path}")
            continue

        mesh = trimesh.load_mesh(mesh_path, process=False)
        if getattr(geom, "scale", None):
            mesh.apply_scale(geom.scale)

        mesh.apply_transform(T_link @ visual.origin)
        mesh_list.append(mesh)
    return mesh_list


def sample_urdf_pcd(urdf_dir: Path, link_name_list: list, robot: URDF, final_pts_num: int = 10000) -> np.ndarray:
    print("Sampling point clouds …")
    all_meshes: List[trimesh.Trimesh] = []
    for link_name in link_name_list:  # type: ignore[attr-defined]
        print(f"Processing link: {link_name}")
        # yourdfpy returns transform from base frame (world) to `link` with cfg applied.
        T_link = robot.get_transform(link_name)  # frame_from defaults to base
        mesh_list = process_visuals(urdf_dir, robot, link_name, T_link)
        if len(mesh_list) > 0:
            all_meshes.extend(mesh_list)
    combined = trimesh.util.concatenate(all_meshes)              # a single TriangleMesh
    points, _ = trimesh.sample.sample_surface(combined, final_pts_num)

    rotate_back = R.from_euler('zyx', [90, 0, -90], degrees=True).as_matrix()
    merged = (rotate_back @ points.T).T
    return merged


def remove_overlay(masks: List[np.ndarray]) -> List[np.ndarray]:
    random_segment_scalar = np.random.rand(masks[0].shape[0], 1, 1) * 10
    part_id_list = []
    tolerance = 1e-7
    full_new_segments = []
    erosion_kernel = np.ones((5, 5), np.uint8) 

    for frame_id, mask in enumerate(masks): # iterate each frame
        random_segments = mask * random_segment_scalar
        blend_segments = np.sum(random_segments, axis=0) # H, W
        current_part_id_array = np.unique(blend_segments)
        current_part_id_array = current_part_id_array[current_part_id_array > 0] # remove 0
        # check new part id
        for current_part_id in current_part_id_array:
            new_id = True
            for exist_layer, old_part_id in enumerate(part_id_list):
                if abs(old_part_id - current_part_id) < tolerance: # find old part id
                    new_id = False
                    break
            if new_id:
                # new_id_list.append(current_part_id)
                part_id_list.append(current_part_id)
        new_segment_list = []
        for layer, part_id in enumerate(part_id_list):
            new_part_segment = (np.abs(blend_segments - part_id) < tolerance)
            new_part_segment = new_part_segment.astype(np.uint8) * 255
            erode_new_part_segment = cv2.erode(new_part_segment, erosion_kernel)
            erode_new_part_segment = (erode_new_part_segment // 255).astype(np.bool_)
            new_segment_list.append(erode_new_part_segment)
        new_segment = np.stack(new_segment_list, axis=0) # current_parts(will change), H, W
        full_new_segments.append(new_segment)
    for frame_id in range(len(full_new_segments)):
        if full_new_segments[frame_id].shape[0] != len(part_id_list):
            padding_matrix = np.zeros((len(part_id_list) - full_new_segments[frame_id].shape[0], full_new_segments[frame_id].shape[1], full_new_segments[frame_id].shape[2]))
            padding_new_segments = np.vstack([full_new_segments[frame_id], padding_matrix])
            full_new_segments[frame_id] = padding_new_segments
    return full_new_segments