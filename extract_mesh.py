import torch
import numpy as np
import nksr
import open3d as o3d
import os
from pycg import vis
import argparse

from data import DataLoader, SimDataLoader, RealDataLoader


def main(args):
    device = torch.device(args.device)
    reconstructor = nksr.Reconstructor(device)

    # init pc
    if args.data_type == "sim":
        data_loader = SimDataLoader(args.view_dir, None, None, None)
    else:
        data_loader = RealDataLoader(args.view_dir, args.preprocess_dir)
    surface_rgb_np, surface_xyz_np = data_loader.load_obj_surface(sample_num=480 * 640)
    surface_rgb = torch.from_numpy(surface_rgb_np).to(device) / 255. # sample_num * 3
    surface_xyz = torch.from_numpy(surface_xyz_np).to(device) # sample_num * 3

    # video first pc
    rgb_list, xyz_list = data_loader.load_rgbd_video()
    xyz = xyz_list[0].reshape(-1, 3)  # H*W, 3
    camera_init = data_loader.load_gt_init_camera_pose_se3()
    xyz = np.dot(xyz, camera_init[:3, :3].T) + camera_init[:3, 3]  # H*W, 3

    o3d_surface = o3d.geometry.PointCloud()
    o3d_surface.points = o3d.utility.Vector3dVector(surface_xyz_np)
    o3d_surface.colors = o3d.utility.Vector3dVector(surface_rgb_np / 255.0)
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
        o3d.io.write_triangle_mesh(f"{args.refinement_results_dir}/surface_mesh.ply", o3d_mesh)
        sample_surface_pcd = o3d_mesh.sample_points_uniformly(number_of_points=10000)
        o3d.io.write_point_cloud(f"{args.refinement_results_dir}/surface_pcd.ply", sample_surface_pcd)
    except:
        print("error encountered when reconstructing surface mesh")
        return
    
    joint_type_list = ["revolute", "prismatic"]
    moving_thresh = 0.7
    dist_thresh = 0.03
    for joint_type in joint_type_list:
        if not os.path.exists(f"{args.refinement_results_dir}/{joint_type}/moving_map.npz"):
            continue
        moving_map = np.load(f"{args.refinement_results_dir}/{joint_type}/moving_map.npz")['a']

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
            o3d.io.write_triangle_mesh(f"{args.refinement_results_dir}/{joint_type}/moving_mesh.ply", moving_o3d_mesh)
            sample_moving_pcd = moving_o3d_mesh.sample_points_uniformly(number_of_points=10000)
            o3d.io.write_point_cloud(f"{args.refinement_results_dir}/{joint_type}/moving_pcd.ply", sample_moving_pcd)
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
            o3d.io.write_triangle_mesh(f"{args.refinement_results_dir}/{joint_type}/static_mesh.ply", static_o3d_mesh)
            sample_static_pcd = static_o3d_mesh.sample_points_uniformly(number_of_points=10000)
            o3d.io.write_point_cloud(f"{args.refinement_results_dir}/{joint_type}/static_pcd.ply", sample_static_pcd)
        except:
            print("error encountered when reconstructing static mesh")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, choices=["sim", "real"], required=True)
    parser.add_argument("--view_dir", type=str, required=True)
    parser.add_argument("--preprocess_dir", type=str, default=None, help="Only required for real data, the directory where the preprocessed data is stored.")
    parser.add_argument("--refinement_results_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    main(args)