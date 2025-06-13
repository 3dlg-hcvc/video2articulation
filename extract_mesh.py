import torch
import numpy as np
import cv2
import nksr
import open3d as o3d
import pickle
import glob
import os
from pycg import vis
import argparse

from utils import depth2xyz, precompute_camera2label, precompute_object2label


def main(args):
    view_dir = args.view_dir
    device = torch.device(args.device)
    reconstructor = nksr.Reconstructor(device)
    intrinsics = np.load(f"{view_dir}/intrinsics.npy")

    joint_data_dir = view_dir[:view_dir.rfind("view_")]
    log_dir = args.results_dir

    # init pc
    actor_pose_path = f"{joint_data_dir}/actor_pose.pkl"
    with open(actor_pose_path, 'rb') as f:
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

    sample_num = min(480 * 640, surface_rgb.shape[0])
    sample_index = np.random.choice(np.arange(surface_rgb.shape[0]), sample_num, replace=False)
    surface_rgb = torch.from_numpy(surface_rgb[sample_index]).to(device) # sample_num * 3
    surface_xyz = torch.from_numpy(surface_xyz[sample_index]).to(device) # sample_num * 3
    surface_segment = torch.from_numpy(surface_segment[sample_index]).to(device) # sample_num * 1

    surface_xyz = surface_xyz[surface_segment]
    surface_rgb = surface_rgb[surface_segment] / 255.

    init_base_pose = obj_pose_dict["actor_6"][0]
    object2label_np = precompute_object2label(init_base_pose)
    object2label = torch.from_numpy(object2label_np).to(device)
    surface_xyz = surface_xyz.to(torch.float64)
    surface_xyz = torch.matmul(surface_xyz, object2label[:3, :3].T) + object2label[:3, 3]
    
    # video pc
    depth_init = np.load(f"{view_dir}/depth/000000.npz")['a']
    xyz = depth2xyz(depth_init, intrinsics) # in opengl coordinate, H, W, 3
    H, W, C = xyz.shape
    bgr = cv2.imread(f"{view_dir}/rgb/000000.jpg")
    rgb_init = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    camera_pose = np.load(f"{view_dir}/camera_pose.npy")
    camera_init = precompute_camera2label(camera_pose[0], object2label_np)
    xyz = np.dot(xyz.reshape(H*W, C), camera_init[:3, :3].T) + camera_init[:3, 3] # transform to label coordinate, H*W, 3

    o3d_surface = o3d.geometry.PointCloud()
    o3d_surface.points = o3d.utility.Vector3dVector(surface_xyz.cpu().numpy())
    o3d_surface.colors = o3d.utility.Vector3dVector(surface_rgb.cpu().numpy())
    o3d_surface.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=80))
    surface_normals = torch.from_numpy(np.asarray(o3d_surface.normals)).to(device)
    surface_xyz = surface_xyz.to(torch.float32)
    surface_rgb = surface_rgb.to(torch.float32)
    surface_normals = surface_normals.to(torch.float32)
    
    try:
        # Note that input_xyz and input_normal are torch tensors of shape [N, 3] and [N, 3] respectively.
        field = reconstructor.reconstruct(surface_xyz, surface_normals, detail_level=1.0)
        # input_color is also a tensor of shape [N, 3]
        field.set_texture_field(nksr.fields.PCNNField(surface_xyz, surface_rgb))
        # Increase the dual mesh's resolution.
        mesh = field.extract_dual_mesh(mise_iter=2)

        # Visualizing (mesh.c is Vx3 array for per-vertex color)
        o3d_mesh = vis.mesh(mesh.v, mesh.f, color=mesh.c)
        # vis.show_3d([vis.mesh(mesh.v, mesh.f, color=mesh.c)])
        o3d.io.write_triangle_mesh(f"{log_dir}/surface_mesh.ply", o3d_mesh)
        sample_surface_pcd = o3d_mesh.sample_points_uniformly(number_of_points=10000)
        o3d.io.write_point_cloud(f"{log_dir}/surface_pcd.ply", sample_surface_pcd)
    except:
        print("error encountered when reconstructing surface mesh")
        return
    
    joint_type_list = ["revolute", "prismatic"]
    moving_thresh = 0.7
    dist_thresh = 0.03
    for joint_type in joint_type_list:
        if not os.path.exists(f"{log_dir}/{joint_type}/moving_map.npz"):
            continue
        moving_map = np.load(f"{log_dir}/{joint_type}/moving_map.npz")['a']

        init_moving_map = moving_map[0] # H, W
        init_moving_map = init_moving_map > moving_thresh
        moving_pcd = xyz[init_moving_map.reshape(-1)]
        sample_moving_index = np.random.choice(np.arange(moving_pcd.shape[0]), min(10000, moving_pcd.shape[0]), replace=False)
        moving_pcd = moving_pcd[sample_moving_index]
        moving_pcd_torch = torch.from_numpy(moving_pcd).to(device).to(torch.float32)
        cdist = torch.cdist(surface_xyz, moving_pcd_torch) # sample_num, moving_pcd_num
        if cdist.shape[1] == 0:
            moving_surface_mask = torch.zeros(surface_xyz.shape[0], dtype=torch.bool).to(device)
        else:
            cdist_min = torch.min(cdist, dim=1)[0]
            moving_surface_mask = cdist_min < dist_thresh
        moving_surface_xyz = surface_xyz[moving_surface_mask]
        moving_surface_rgb = surface_rgb[moving_surface_mask]
        moving_surface_normals = surface_normals[moving_surface_mask]
        try:
            field = reconstructor.reconstruct(moving_surface_xyz, moving_surface_normals, detail_level=1.0)
            # input_color is also a tensor of shape [N, 3]
            field.set_texture_field(nksr.fields.PCNNField(moving_surface_xyz, moving_surface_rgb))
            # Increase the dual mesh's resolution.
            moving_mesh = field.extract_dual_mesh(mise_iter=2)
            # Visualizing (mesh.c is Vx3 array for per-vertex color)
            moving_o3d_mesh = vis.mesh(moving_mesh.v, moving_mesh.f, color=moving_mesh.c)
            # vis.show_3d([vis.mesh(mesh.v, mesh.f, color=mesh.c)])
            o3d.io.write_triangle_mesh(f"{log_dir}/{joint_type}/moving_mesh.ply", moving_o3d_mesh)
            # o3d.io.write_triangle_mesh(f"{log_dir}/moving_mesh_dense.ply", moving_o3d_mesh)
            sample_moving_pcd = moving_o3d_mesh.sample_points_uniformly(number_of_points=10000)
            # o3d.io.write_point_cloud(f"{log_dir}/{joint_type}/moving_pcd.ply", sample_moving_pcd)
            o3d.io.write_point_cloud(f"{log_dir}/{joint_type}/moving_pcd.ply", sample_moving_pcd)
        except:
            print("error encountered when reconstructing moving mesh")
            continue

        static_surface_xyz = surface_xyz[~moving_surface_mask]
        static_surface_rgb = surface_rgb[~moving_surface_mask]
        static_surface_normals = surface_normals[~moving_surface_mask]
        try:
            field = reconstructor.reconstruct(static_surface_xyz, static_surface_normals, detail_level=1.0)
            # input_color is also a tensor of shape [N, 3]
            field.set_texture_field(nksr.fields.PCNNField(static_surface_xyz, static_surface_rgb))
            # Increase the dual mesh's resolution.
            static_mesh = field.extract_dual_mesh(mise_iter=2)
            # Visualizing (mesh.c is Vx3 array for per-vertex color)
            static_o3d_mesh = vis.mesh(static_mesh.v, static_mesh.f, color=static_mesh.c)
            # vis.show_3d([vis.mesh(mesh.v, mesh.f, color=mesh.c)])
            o3d.io.write_triangle_mesh(f"{log_dir}/{joint_type}/static_mesh.ply", static_o3d_mesh)
            # o3d.io.write_triangle_mesh(f"{log_dir}/static_mesh_dense.ply", static_o3d_mesh)
            sample_static_pcd = static_o3d_mesh.sample_points_uniformly(number_of_points=10000)
            # o3d.io.write_point_cloud(f"{log_dir}/{joint_type}/static_pcd.ply", sample_static_pcd)
            o3d.io.write_point_cloud(f"{log_dir}/{joint_type}/static_pcd.ply", sample_static_pcd)
        except:
            print("error encountered when reconstructing static mesh")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    main(args)