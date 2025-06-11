import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import pickle
import random


def set_seed(seed: int):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


# def inverse_transformation(T: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
#     """
#     Compute the inverse of an SE(3) transformation matrix.

#     Parameters:
#         T (numpy.ndarray | torch.Tensor): Bx4x4 transformation matrix.

#     Returns:
#         numpy.ndarray: Inverse of the transformation matrix.
#     """
#     # Extract the rotation matrix (R) and translation vector (t)
#     R = T[:3, :3]
#     t = T[:3, 3]

#     # Compute the inverse
#     R_inv = R.T  # Transpose of the rotation matrix
#     t_inv = -np.dot(R_inv, t)  # Negated product of R_inv and t

#     # Construct the inverse transformation matrix
#     T_inv = np.eye(4)
#     T_inv[:3, :3] = R_inv
#     T_inv[:3, 3] = t_inv

#     return T_inv


def inverse_transformation(T: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """
    Invert a 4x4 SE(3) transformation matrix or a batch of such matrices.

    Args:
        T: (4, 4) or (N, 4, 4) SE(3) transformation matrix.
           Can be a numpy.ndarray or torch.Tensor.

    Returns:
        Inverse of the transformation(s), same type and shape as input.
    """
    # Determine if the input is a torch tensor or numpy array
    is_torch = isinstance(T, torch.Tensor)

    # Extract rotation and translation
    R = T[..., :3, :3]  # shape (..., 3, 3)
    t = T[..., :3, 3:]  # shape (..., 3, 1)

    # Compute inverse rotation and translation
    R_inv = R.transpose(-1, -2)  # Transpose of rotation matrix
    t_inv = -R_inv @ t           # Inverse translation

    # Assemble inverse matrix
    T_inv = T.clone() if is_torch else T.copy()
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3] = t_inv.squeeze(-1)
    T_inv[..., 3, :] = 0
    T_inv[..., 3, 3] = 1

    return T_inv


def depth2xyz(depth_image: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    # Get the shape of the depth image
    H, W = depth_image.shape

    # Create meshgrid for pixel coordinates (u, v)
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Flatten the grid to a 1D array of pixel coordinates
    u = u.flatten()
    v = v.flatten()

    # Flatten the depth image to a 1D array of depth values
    if depth_image.dtype == np.uint16:
        depth = depth_image.flatten() / 1000.0
    else:
        depth = depth_image.flatten()

    # Camera intrinsic matrix (3x3)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Calculate the 3D coordinates (x, y, z) from depth
    # Use the formula:
    #   X = (u - cx) * depth / fx
    #   Y = (v - cy) * depth / fy
    #   Z = depth
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    # Stack the x, y, z values into a 3D point cloud
    point_cloud = np.vstack((x, y, z)).T

    point_cloud = point_cloud * np.array([1, -1, -1])

    # Reshape the point cloud to the original depth image shape [H, W, 3]
    point_cloud = point_cloud.reshape(H, W, 3)

    return point_cloud


def estimate_se3_transformation(target_xyz: np.ndarray, source_xyz: np.ndarray) -> np.ndarray:
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_xyz)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_xyz)
    correspondences = np.arange(source_xyz.shape[0])
    correspondences = np.vstack([correspondences, correspondences], dtype=np.int32).T
    p2p_registration = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False)
    source2target = p2p_registration.compute_transformation(source_pcd, target_pcd, o3d.utility.Vector2iVector(correspondences))
    return source2target


def find_movable_part(actor_pose_path: str) -> int:
    with open(actor_pose_path, 'rb') as f:
        actor_pose_dict = pickle.load(f)
    moving_part_id = -1
    max_diff = -1
    for actor in actor_pose_dict.keys():
        init_pose = actor_pose_dict[actor][0]
        end_pose = actor_pose_dict[actor][-1]
        diff = np.linalg.norm(end_pose - init_pose)
        if max_diff < diff:
            moving_part_id = int(actor[6:])
            max_diff = diff

    return moving_part_id


def precompute_camera2label(gt_camera_pose_path: str, actor_pose_path: str, index: int = 0) -> np.ndarray:
    camera_pose = np.load(gt_camera_pose_path)
    camera2object_rotation = R.from_quat(camera_pose[index, 3:], scalar_first=True).as_matrix()
    camera2object_translation = camera_pose[index, :3]
    camera2object_rotation = camera2object_rotation[:, [1, 2, 0]] * [[-1, 1, -1]]
    camera2object = np.eye(4)
    camera2object[:3, :3] = camera2object_rotation
    camera2object[:3, 3] = camera2object_translation

    object2label = precompute_object2label(actor_pose_path)

    camera2label = np.dot(object2label, camera2object)
    return camera2label


def precompute_object2label(actor_pose_path: str) -> np.ndarray:
    with open(actor_pose_path, 'rb') as f:
        obj_pose_dict = pickle.load(f)
    init_base_pose = obj_pose_dict["actor_6"][0]
    actor_translation = init_base_pose[:3]
    actor_rotation = R.from_quat(init_base_pose[3:], scalar_first=True).as_matrix()
    origin2object = np.eye(4)
    origin2object[:3, :3] = actor_rotation
    origin2object[:3, 3] = actor_translation
    object2origin = inverse_transformation(origin2object)

    origin2label_rotation = R.from_euler('zyx', [90, 0, -90], degrees=True).as_matrix()
    origin2label = np.eye(4)
    origin2label[:3, :3] = origin2label_rotation

    object2label = np.dot(origin2label, object2origin)
    return object2label