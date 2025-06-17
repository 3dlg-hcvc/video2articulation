from kornia.feature import LoFTR
import numpy as np
import cv2
import torch
from scipy.spatial.transform import Rotation as R
import glob
import json
import argparse
from typing import Tuple

from utils import estimate_se3_transformation
from data import RealDataLoader


def estimate_camera_se3(base_kp: np.ndarray, curr_kp: np.ndarray, RANSAC: bool) -> Tuple[np.ndarray, np.ndarray]:
    curr2base = None
    inlier = None
    if RANSAC:
        k = 50
        inlier_thresh = 1e-1
        d = int(base_kp.shape[0] * 0.4)
        best_se3 = None
        best_error = 10000
        best_inlier = None
        inlier_list = []
        se3_list = []
        inlier_index_list = []
        for _ in range(k):
            init_sample = np.random.choice(base_kp.shape[0], min(10, base_kp.shape[0]), replace=False)
            init_kp1 = base_kp[init_sample]
            init_kp2 = curr_kp[init_sample]

            se3 = estimate_se3_transformation(init_kp1, init_kp2)
            se3_list.append(se3)
            rotation = se3[:3, :3]
            translation = se3[:3, 3]

            transform_kp2 = curr_kp @ rotation.T + translation
            dist = np.linalg.norm((base_kp - transform_kp2), axis=1)
            inlier = np.nonzero(dist < inlier_thresh)[0]
            inlier_list.append(inlier.shape[0])
            inlier_index_list.append(inlier)
            if inlier.shape[0] > d:
                se3 = estimate_se3_transformation(base_kp[inlier], curr_kp[inlier])
                se3_list[-1] = se3
                rotation = se3[:3, :3]
                translation = se3[:3, 3]

                transform_inlier_kp2 = curr_kp[inlier] @ rotation.T + translation
                this_error = np.mean((base_kp[inlier] - transform_inlier_kp2) ** 2)
                if this_error < best_error:
                    best_se3 = se3
                    best_error = this_error
                    best_inlier = inlier
        if best_se3 is None:
            print("RANSAC fail!")
            max_inlier_index = inlier_list.index(max(inlier_list))
            best_se3 = se3_list[max_inlier_index]
            best_inlier = inlier_index_list[max_inlier_index]
        else:
            print("RANSAC success!")
        curr2base = best_se3
        inlier = best_inlier
    else:
        curr2base = estimate_se3_transformation(base_kp, curr_kp)
        inlier = np.arange(base_kp.shape[0])
    return curr2base, inlier


def compute_match(img1_raw: np.ndarray, img2_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # img1_raw = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img1_gray = cv2.cvtColor(img1_raw, cv2.COLOR_RGB2GRAY)
    img1_torch = torch.Tensor(img1_gray).cuda() / 255.
    img1_torch = torch.reshape(img1_torch, (1, 1, img1_torch.shape[0], img1_torch.shape[1]))
    # img2_raw = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    img2_gray = cv2.cvtColor(img2_raw, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.rotate(img2_gray, cv2.ROTATE_90_CLOCKWISE)
    # print(img2_raw.shape)
    img2_torch = torch.Tensor(img2_gray).cuda() / 255.
    img2_torch = torch.reshape(img2_torch, (1, 1, img2_torch.shape[0], img2_torch.shape[1]))

    input = {"image0": img1_torch, "image1": img2_torch}
    matcher = LoFTR(pretrained='indoor').cuda()
    correspondences_dict = matcher(input)

    mkpts0 = correspondences_dict['keypoints0'].cpu().numpy()
    mkpts1 = correspondences_dict['keypoints1'].cpu().numpy()
    mconf = correspondences_dict['confidence'].cpu().numpy()
    return mkpts0, mkpts1, mconf


def scan2mesh(xyz_img: np.ndarray, camera_pose: np.ndarray, mesh_align: np.ndarray) -> np.ndarray:
    xyz_img = xyz_img * np.array([1, -1, -1])
    xyz_img = xyz_img @ camera_pose[:3, :3].T + camera_pose[:3, 3]
    xyz_img = xyz_img @ mesh_align[:3, :3].T + mesh_align[:3, 3]
    rot90 = R.from_euler('x', 90, degrees=True).as_matrix()
    xyz_world = xyz_img @ rot90.T
    return xyz_world


def main(args):
    data_loader = RealDataLoader(args.view_dir, args.preprocess_dir)

    rgb_list, xyz_list = data_loader.load_rgbd_video()
    init_img = rgb_list[0]
    init_xyz = xyz_list[0]
    surface_img_list, surface_xyz_list, surface_cameras_config_paths = data_loader.load_surface_rgbd_cameras()

    best_mkpts0, best_mkpts1 = None, None
    max_match_num = 0
    best_camera_id = 0
    for id, surface_img in enumerate(surface_img_list):
        mkpts0, mkpts1, mconf = compute_match(init_img, surface_img)
        match_mask = mconf > 0.99
        mkpts0 = mkpts0[match_mask].astype(np.uint32)
        mkpts1 = mkpts1[match_mask].astype(np.uint32)
        if np.sum(match_mask) > max_match_num:
            best_mkpts0, best_mkpts1, max_match_num = mkpts0, mkpts1, np.sum(match_mask)
            best_camera_id = id

    best_surface_xyz = surface_xyz_list[best_camera_id]
    best_surface_camera =  surface_cameras_config_paths[best_camera_id]
    with open(best_surface_camera, 'r') as f:
        surface_img_meta = json.load(f)
    surface_camera_pose = np.array([[surface_img_meta["t_00"], surface_img_meta["t_01"], surface_img_meta["t_02"], surface_img_meta["t_03"]],
                                    [surface_img_meta["t_10"], surface_img_meta["t_11"], surface_img_meta["t_12"], surface_img_meta["t_13"]],
                                    [surface_img_meta["t_20"], surface_img_meta["t_21"], surface_img_meta["t_22"], surface_img_meta["t_23"]],
                                    [0, 0, 0, 1]])
    mesh_align = data_loader.mesh_align
    surface_xyz_world = scan2mesh(best_surface_xyz, surface_camera_pose, mesh_align)
    surface_xyz_world = np.rot90(surface_xyz_world, axes=(1, 0))
    static_kp1 = surface_xyz_world[best_mkpts1[:, 1], best_mkpts1[:, 0]]
    static_kp0 = init_xyz[best_mkpts0[:, 1], best_mkpts0[:, 0]]
    video_camera2world, _ = estimate_camera_se3(static_kp1, static_kp0, RANSAC=True)
    np.save(f"{args.preprocess_dir}/debug_cam2world.npy", video_camera2world)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view_dir", type=str, required=True)
    parser.add_argument("--preprocess_dir", type=str, required=True)
    args = parser.parse_args()
    main(args)