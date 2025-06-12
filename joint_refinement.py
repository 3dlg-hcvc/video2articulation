import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
from pytorch3d.loss import chamfer_distance
from pytorch3d.transforms import axis_angle_to_matrix, quaternion_to_matrix
from scipy.spatial.transform import Rotation as R
import wandb

import random
import pickle
import json
import os
import shutil
import time
import glob
from tqdm import tqdm
import argparse
import yaml
from typing import Tuple, List

from utils import set_seed, find_movable_part, precompute_object2label, precompute_camera2label, depth2xyz


class BundleAdjustment(torch.nn.Module):
    def __init__(self, joint_data_dir: str, preprocess_dir: str, prediction_dir: str, opt_view: int, mask_type: str, 
                 joint_type: str, lr: float, loss_func: str, opt_steps: int, log_dir: str, 
                 device: torch.device, seed: int, vis_pcd: bool = False):
        super(BundleAdjustment, self).__init__()
        set_seed(seed)
        self.device = device
        # init pc
        self.actor_pose_path = f"{joint_data_dir}/actor_pose.pkl"
        with open(f"{joint_data_dir}/actor_pose.pkl", 'rb') as f:
            obj_pose_dict = pickle.load(f)
        actor_list = []
        for actor_name in obj_pose_dict.keys():
            actor_list.append(int(actor_name[6:]))
        moving_part_id = find_movable_part(obj_pose_dict)
        
        surface_dir = f"{joint_data_dir}/view_init"
        surface_img = []
        surface_xyz = []
        surface_mask = []
        surface_dynamic_mask_list = []
        for i in range(24):
            img = cv2.imread("{}/rgb/{}".format(surface_dir, "%06d.png" % i))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            surface_img.append(img)
            xyz = np.load("{}/xyz/{}".format(surface_dir, "%06d.npz" % i))['a']
            surface_xyz.append(xyz)
            segment = np.load("{}/segment/{}".format(surface_dir, "%06d.npz" % i))['a']
            mask = segment == -1
            surface_dynamic_mask = segment == moving_part_id
            for actor_id in actor_list:
                mask_id = segment == actor_id
                mask = np.logical_or(mask, mask_id)
            surface_mask.append(mask)
            surface_dynamic_mask_list.append(surface_dynamic_mask)
        self.surface_rgb = np.stack(surface_img).reshape(-1, 3)
        self.surface_xyz = np.stack(surface_xyz).reshape(-1, 3)
        self.surface_segment = np.stack(surface_mask).flatten()
        self.surface_dynamic_segment = np.stack(surface_dynamic_mask_list).flatten()

        sample_num = min(480 * 64, self.surface_rgb.shape[0])
        sample_index = np.random.choice(np.arange(self.surface_rgb.shape[0]), sample_num, replace=False)
        self.surface_rgb = torch.from_numpy(self.surface_rgb[sample_index]).to(device) # sample_num * 3
        self.surface_xyz = torch.from_numpy(self.surface_xyz[sample_index]).to(device) # sample_num * 3
        self.surface_segment = torch.from_numpy(self.surface_segment[sample_index]).to(device) # sample_num * 1
        self.surface_dynamic_segment = torch.from_numpy(self.surface_dynamic_segment[sample_index]).to(device) # sample_num * 1
        self.surface_static_segment = torch.logical_and(self.surface_segment, ~self.surface_dynamic_segment) # sample_num * 1
        
        self.surface_static_xyz = self.surface_xyz[self.surface_static_segment].to(torch.float64)
        self.surface_dynamic_xyz = self.surface_xyz[self.surface_dynamic_segment].to(torch.float64)
        self.surface_xyz = self.surface_xyz[self.surface_segment]
        self.surface_rgb = self.surface_rgb[self.surface_segment]

        init_base_pose = obj_pose_dict["actor_6"][0]
        object2label_np = precompute_object2label(init_base_pose)
        object2label = torch.from_numpy(object2label_np).to(torch.float64).to(device) # 4, 4
        self.surface_xyz = self.surface_xyz.to(torch.float64)
        self.surface_xyz = torch.matmul(self.surface_xyz, object2label[:3, :3].T) + object2label[:3, 3]
        self.surface_static_xyz = torch.matmul(self.surface_static_xyz, object2label[:3, :3].T) + object2label[:3, 3]
        self.surface_dynamic_xyz = torch.matmul(self.surface_dynamic_xyz, object2label[:3, :3].T) + object2label[:3, 3]
        
        view_dir = f"{joint_data_dir}/view_{opt_view}"

        # optimize parameter
        ## camera pose
        camera_pose_list = glob.glob(f"{prediction_dir}/coarse_prediction/{mask_type}/{seed}/cam_pose/*.npy")
        self.valid = True
        if len(camera_pose_list) == 0:
            self.valid = False
            return
        camera_pose_list.sort(key=lambda x: int(x[x.rfind('_') + 1:-4]))
        camera_pose = []
        for camera_pose_file in camera_pose_list:
            camera_extrinsics = np.load(camera_pose_file)
            camera_translation = camera_extrinsics[:3, 3]
            camera_rotation = R.from_matrix(camera_extrinsics[:3, :3]).as_quat(scalar_first=True)
            camera_pose.append(torch.from_numpy(np.concatenate([camera_rotation, camera_translation])))
        self.camera_pose = torch.nn.Parameter(torch.stack(camera_pose).to(device), requires_grad=True) # N, 7  first 4 quaternion, last 3 translation
        
        ## joint axis, joint position, and joint state
        self.joint_axis = torch.nn.Parameter(torch.from_numpy(np.load(f"{prediction_dir}/coarse_prediction/{mask_type}/{seed}/{joint_type}/joint_axis.npy")).to(device), requires_grad=True) # 3,
        self.joint_pos = torch.nn.Parameter(torch.from_numpy(np.load(f"{prediction_dir}/coarse_prediction/{mask_type}/{seed}/{joint_type}/joint_pos.npy")).to(device), requires_grad=True) # 3,
        self.joint_state = torch.nn.Parameter(torch.from_numpy(np.load(f"{prediction_dir}/coarse_prediction/{mask_type}/{seed}/{joint_type}/joint_value.npy")).to(device), requires_grad=True) # N,
        if torch.any(torch.isnan(self.joint_state)):
            del self.joint_state
            if joint_type == "revolute":
                self.joint_state = torch.nn.Parameter(torch.linspace(0, torch.pi / 2, self.camera_pose.shape[0], device=device), requires_grad=True) # N,
            elif joint_type == "prismatic":
                self.joint_state = torch.nn.Parameter(torch.linspace(0, 0.1, self.camera_pose.shape[0], device=device), requires_grad=True) # N,
        self.joint_type = joint_type
        
        ## moving map
        # gt moving map
        sample_rgb_dir = f"{view_dir}/sample_rgb"
        segment_dir = f"{view_dir}/segment"
        img_list = os.listdir(sample_rgb_dir)
        img_list.sort()
        sample_rgb_index = [int(file_name[:-4]) for file_name in os.listdir(f"{view_dir}/sample_rgb/")]
        sample_rgb_index.sort()
        
        gt_moving_map = []
        obj_mask_list = []
        for i in range(len(img_list)):
            segment = np.load(f"{segment_dir}/{img_list[i][:-4]}.npz")['a']
            dynamic_mask = segment == moving_part_id
            gt_moving_map.append(torch.from_numpy(dynamic_mask))
            obj_mask = segment == -1
            for actor_id in actor_list:
                obj_mask_id = segment == actor_id
                obj_mask = np.logical_or(obj_mask, obj_mask_id)
            obj_mask_list.append(obj_mask)
        self.gt_moving_map_bool = torch.stack(gt_moving_map).to(device) # N, H, W
        # obj mask
        self.obj_mask_bool = torch.from_numpy(np.stack(obj_mask_list)).to(device) # N, H, W
        self.obj_mask = torch.from_numpy(np.stack(obj_mask_list)).to(torch.float64).to(device) # N, H, W

        # video pc
        self.intrinsics = np.load(f"{view_dir}/intrinsics.npy")
        self.intrinsics_torch = torch.from_numpy(self.intrinsics).to(device)
        self.intrinsics_torch = self.intrinsics_torch.to(torch.float64)
        depth_list = glob.glob(f"{view_dir}/depth/*.npz")
        depth_list.sort()
        xyz = []
        rgb_list = glob.glob(f"{view_dir}/rgb/*.jpg")
        rgb_list.sort()
        rgb = []
        for i in sample_rgb_index:
            depth = np.load(depth_list[i])['a']
            xyz.append(depth2xyz(depth, self.intrinsics)) # in opengl coordinate
            bgr = cv2.imread(rgb_list[i])
            rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            
        self.xyz = torch.from_numpy(np.stack(xyz)).to(device) # N, H, W, 3
        self.rgb = torch.from_numpy(np.stack(rgb)).to(device) # N, H, W, 3
        N, H, W, C = self.rgb.shape
        # part segments map
        segments_map_list = glob.glob(f"{preprocess_dir}/video_segment_reverse/small/final-output/*.npz")
        segments_map_list.sort(key=lambda x: int(x[x.rfind('_') + 1:-4]), reverse=True)
        segments_map_list = [segments_map_list[i] for i in sample_rgb_index]
        origin_segments_list = []
        for segments_map in segments_map_list:
            origin_segments_list.append(np.load(segments_map)['a'][:, 0, :, :])
        non_overlap_segments_list = self.remove_overlay(origin_segments_list)
        initial_segments = non_overlap_segments_list[0]
        self.old_segment_id_list = []
        initial_obj_mask = self.obj_mask_bool[0].cpu().numpy()
        self.part_segments_list = []
        old_part_segments_list = []
        for segment_id in range(initial_segments.shape[0]):
            part_segment0 = initial_segments[segment_id]
            part_occupation = np.logical_and(part_segment0, initial_obj_mask)
            if np.sum(part_occupation) > 100:
                self.old_segment_id_list.append(segment_id)
        for frame_id, segments_map in enumerate(segments_map_list):
            part_segments = non_overlap_segments_list[frame_id]
            obj_segments = self.obj_mask_bool[frame_id].cpu().numpy() # H, W
            part_segments = np.logical_and(part_segments[self.old_segment_id_list], obj_segments[np.newaxis, ...])
            self.part_segments_list.append(part_segments) # old_parts, H, W
            old_part_segments = np.zeros_like(obj_segments, dtype=np.bool_)
            for part_id in range(part_segments.shape[0]):
                old_part_segments = np.logical_or(old_part_segments, part_segments[part_id])
            old_part_segments_list.append(torch.from_numpy(old_part_segments).to(self.device)) # H, W
        self.old_part_segments_list = torch.stack(old_part_segments_list).to(self.device) # N, H, W

        part_segments_full = np.stack(self.part_segments_list) # numpy N, old_parts, H, W
        self.part_segments_full = torch.from_numpy(part_segments_full).to(torch.float64).to(self.device) # N, old_parts, H, W

        if mask_type == "monst3r":
            moving_map_list = glob.glob(f"{preprocess_dir}/monst3r/dynamic_mask_*.png")
            moving_map_list.sort(key=lambda x: int(x[x.rfind('_') + 1:-4]))
            # moving_map_list = [moving_map_list[i] for i in opt_index]
            part_moving_occupation = np.zeros((len(self.old_segment_id_list),))
            moving_map = []
            for frame_id, moving_map_file in enumerate(moving_map_list):
                pred_mask_img = Image.open(moving_map_file).convert('L')
                pred_mask_img = pred_mask_img.resize((640, 480), Image.BICUBIC)
                pred_mask = np.array(pred_mask_img, dtype=np.float64)
                pred_mask = (pred_mask / 255.)
                moving_map.append(torch.from_numpy(pred_mask))
                pred_mask_bool = np.round(pred_mask).astype(np.bool_)
                part_segments = self.part_segments_list[frame_id]
                part_moving_occupation += np.sum(np.logical_and(part_segments, pred_mask_bool[np.newaxis, ...]), axis=(1, 2))
            part_moving_occupation = part_moving_occupation / np.sum(part_segments_full, axis=(0, 2, 3))
            self.moving_map_vec = torch.nn.Parameter(torch.from_numpy(part_moving_occupation).to(device), requires_grad=True) # old_parts,
        elif mask_type == "gt":
            self.moving_map = self.gt_moving_map_bool.clone().to(torch.float64)
        
        ## optimizer
        if mask_type == "monst3r":
            optimize_params = [{"params": (self.camera_pose, self.joint_axis, self.joint_pos, self.joint_state, self.moving_map_vec)}]
        elif mask_type == "gt":
            optimize_params = [{"params": (self.camera_pose, self.joint_axis, self.joint_pos, self.joint_state)}]
        self.optimizer = torch.optim.Adam(optimize_params, lr=lr)
        self.steps = opt_steps
        self.loss_func = loss_func
        self.current_step = 0
        self.lr = lr
        self.mask_type = mask_type
        self.vis_pcd = vis_pcd
        
        gt_camera_pose = np.load(f"{view_dir}/camera_pose.npy")
        self.gt_camera_se3 = []
        for i in sample_rgb_index:
            gt_camera_se3_i= precompute_camera2label(gt_camera_pose[i], object2label_np)
            self.gt_camera_se3.append(gt_camera_se3_i)
        self.gt_camera_se3 = np.stack(self.gt_camera_se3) # numpy N, 4, 4

        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, opt_steps)

        self.train = True
        self.best_joint_axis = self.joint_axis.detach().cpu().numpy()
        self.best_joint_pos = self.joint_pos.detach().cpu().numpy()
        self.best_joint_state = self.joint_state.detach().cpu().numpy()
        self.best_camera_poses = self.camera_pose.detach().cpu().numpy()
        self.best_moving_vectors = self.moving_map_vec.detach().cpu().numpy()
        self.best_loss = 100
        self.log_dir = f"{log_dir}/{loss_func}/{seed}/{self.joint_type}"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)


    def dump_configuration(self,):
        config_dict = {"loss_function": self.loss_func, 
                       "mask_type": self.mask_type, 
                       "opt_steps": self.steps, 
                       "lr": self.lr}
        with open(f"{self.log_dir}/opt_config.yaml", "w") as f:
            yaml.dump(config_dict, f)


    def remove_overlay(self, masks: List[np.ndarray]) -> List[np.ndarray]:
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
    

    def chamfer_loss(self,) -> torch.Tensor:
        N, H, W, C = self.xyz.shape
        if self.mask_type == "monst3r":
            full_moving_map = self.part_segments_full * self.moving_map_vec.reshape(1, self.part_segments_full.shape[1], 1, 1) # N, old_parts, H, W
            moving_map = torch.sum(full_moving_map, dim=1) # N, W, H
            moving_map = (moving_map - torch.amin(moving_map, dim=(1, 2), keepdim=True)) / (torch.amax(moving_map, dim=(1, 2), keepdim=True) - torch.amin(moving_map, dim=(1, 2), keepdim=True) + 1e-12)
        else:
            moving_map = self.moving_map
        # static chamfer distance
        camera_extrinsics = torch.eye(4, dtype=torch.float64).repeat(N, 1, 1).to(self.device)
        camera_rotations = quaternion_to_matrix(self.camera_pose[:, :4])
        camera_extrinsics[:, :3, :3] = camera_rotations
        camera_extrinsics[:, :3, 3] = self.camera_pose[:, 4:]
        cam_transformed_xyz = torch.matmul(self.xyz.reshape(N, H*W, C), camera_extrinsics[:, :3, :3].permute(0, 2, 1)) + camera_extrinsics[:, :3, 3].reshape(N, 1, 3) # N, H*W, 3

        static_chamfer_loss = torch.zeros((cam_transformed_xyz.shape[0],))
        for b in range(cam_transformed_xyz.shape[0]):
            obj_mask = self.obj_mask_bool[b].reshape(H*W)
            old_part_segment = self.old_part_segments_list[b].reshape(H*W)
            computable_mask = torch.logical_and(obj_mask, old_part_segment)
            filter_cam_transformed_xyz = cam_transformed_xyz[b, computable_mask, :] # compute_mask_num, 3
            random_sample_num = filter_cam_transformed_xyz.shape[0] // 2
            random_sample_index = torch.randint(0, max(filter_cam_transformed_xyz.shape[0], 1), (random_sample_num,), device=self.device)
            random_filter_cam_transformed_xyz = filter_cam_transformed_xyz[random_sample_index, :] # random_sample_num, 3
            if self.mask_type == "gt":
                static_chamfer_dist_b, _ = chamfer_distance(random_filter_cam_transformed_xyz[None, ...], self.surface_static_xyz[None, ...], 
                                                            batch_reduction=None, point_reduction=None, single_directional=True, ) # 1, mask_num
            else:
                static_chamfer_dist_b, _ = chamfer_distance(random_filter_cam_transformed_xyz[None, ...], self.surface_xyz[None, ...], 
                                                            batch_reduction=None, point_reduction=None, single_directional=True, ) # 1, mask_num
            norm_moving_map_b = moving_map[b].reshape(1, H*W)
            filter_norm_moving_map = norm_moving_map_b[:, computable_mask] # 1, compute_mask_num
            random_filter_norm_moving_map = filter_norm_moving_map[:, random_sample_index] # 1, random_sample_num
            if self.loss_func == "hausdorff":
                weighted_static_chamfer_dist_b = torch.max((1 - random_filter_norm_moving_map) * static_chamfer_dist_b)
            else:
                weighted_static_chamfer_dist_b = torch.mean((1 - random_filter_norm_moving_map) * static_chamfer_dist_b)
            if torch.isnan(weighted_static_chamfer_dist_b):
                static_chamfer_loss[b] = 0
            else:
                static_chamfer_loss[b] = weighted_static_chamfer_dist_b
        # weighted_static_chamfer_dist = self.mask * (1 - norm_moving_map) * (static_chamfer_dist.reshape(N, H, W)) # N, H, W
        # static_chamfer_loss = torch.mean(weighted_static_chamfer_dist, dim=(1, 2)) # N,

        # dynamic chamfer distance
        joint_axis_norm = F.normalize(self.joint_axis.reshape(1, 3))
        if self.joint_type == "revolute":
            rot_vec = joint_axis_norm.repeat(N, 1) * self.joint_state.reshape(N, 1) # N, 3
            rotations = axis_angle_to_matrix(rot_vec) # N, 3, 3
            translations = torch.matmul((torch.eye(3).repeat(N, 1, 1).to(self.device) - rotations), self.joint_pos) # N, 3
        elif self.joint_type == "prismatic":
            rotations = torch.eye(3, dtype=torch.float64, device=self.device).repeat(N, 1, 1) # N, 3, 3
            translations = joint_axis_norm.repeat(N, 1) * self.joint_state.reshape(N, 1) # N, 3
        joint_transformed_xyz = torch.matmul(cam_transformed_xyz, rotations.permute(0, 2, 1)) + translations.reshape(N, 1, 3) # N, H*W, 3
        # np.savetxt("joint_transformed_xyz.xyz", joint_transformed_xyz[-1].detach().cpu().numpy())
        dynamic_chamfer_loss = torch.zeros_like(static_chamfer_loss)
        for b in range(joint_transformed_xyz.shape[0]):
            obj_mask = self.obj_mask_bool[b].reshape(H*W)
            old_part_segment = self.old_part_segments_list[b].reshape(H*W)
            computable_mask = torch.logical_and(obj_mask, old_part_segment)
            filter_joint_transformed_xyz = joint_transformed_xyz[b, computable_mask, :] # compute_mask_num, 3
            random_sample_num = filter_joint_transformed_xyz.shape[0] // 2
            random_sample_index = torch.randint(0, max(filter_joint_transformed_xyz.shape[0], 1), (random_sample_num,), device=self.device)
            random_filter_joint_transformed_xyz = filter_joint_transformed_xyz[random_sample_index, :] # random_sample_num, 3
            if self.mask_type == "gt":
                dynamic_chamfer_dist_b, _ = chamfer_distance(random_filter_joint_transformed_xyz[None, ...], self.surface_dynamic_xyz[None, ...], 
                                                             batch_reduction=None, point_reduction=None, single_directional=True, ) # 1, mask_num
            else:
                dynamic_chamfer_dist_b, _ = chamfer_distance(random_filter_joint_transformed_xyz[None, ...], self.surface_xyz[None, ...], 
                                                             batch_reduction=None, point_reduction=None, single_directional=True, ) # 1, mask_num
            norm_moving_map_b = moving_map[b].reshape(1, H*W)
            filter_norm_moving_map = norm_moving_map_b[:, computable_mask] # 1, compute_mask_num
            random_filter_norm_moving_map = filter_norm_moving_map[:, random_sample_index] # 1, random_sample_num
            if self.loss_func == "hausdorff":
                weighted_dynamic_chamfer_dist_b = torch.max(random_filter_norm_moving_map * dynamic_chamfer_dist_b)
            else:
                weighted_dynamic_chamfer_dist_b = torch.mean(random_filter_norm_moving_map * dynamic_chamfer_dist_b)
            if torch.isnan(weighted_dynamic_chamfer_dist_b):
                dynamic_chamfer_loss[b] = 0
            else:
                dynamic_chamfer_loss[b] = weighted_dynamic_chamfer_dist_b
        # weighted_dynamic_chamfer_dist = self.mask * norm_moving_map * (dynamic_chamfer_dist.reshape(N, H, W)) # N, H, W
        # dynamic_chamfer_loss = torch.mean(weighted_dynamic_chamfer_dist, dim=(1, 2)) # N,

        if self.train:
            wandb.log({"Train/static chamfer loss": static_chamfer_loss.detach().mean().cpu().item(), 
                       "Train/dynamic chamfer loss": dynamic_chamfer_loss.detach().mean().cpu().item(),
                       "Train/chamfer loss": static_chamfer_loss.detach().mean().cpu().item() + dynamic_chamfer_loss.detach().mean().cpu().item()})

        return static_chamfer_loss + dynamic_chamfer_loss
    

    def eval(self,) -> torch.Tensor:
        self.train = False
        with torch.no_grad():
            chamfer_loss = self.chamfer_loss()
            loss = torch.mean(chamfer_loss)
        self.train = True
        return loss


    def optimize_adam(self, gt_joint_type: str, gt_joint_axis: np.ndarray, gt_joint_pos: np.ndarray, gt_joint_value: np.ndarray):
        tbar = tqdm(range(self.steps))
        for i, _ in enumerate(tbar):
            self.current_step = i

            start_time = time.time()
            self.optimizer.zero_grad()
            
            chamfer_loss = self.chamfer_loss()
            loss = torch.mean(chamfer_loss)

            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            end_time = time.time()

            eval_loss = loss.detach()
            if eval_loss.detach().cpu().item() < self.best_loss:
                self.best_joint_axis = self.joint_axis.detach().cpu().numpy()
                self.best_joint_pos = self.joint_pos.detach().cpu().numpy()
                self.best_joint_state = self.joint_state.detach().cpu().numpy()
                self.best_camera_poses = self.camera_pose.detach().cpu().numpy()
                if self.mask_type == "monst3r":
                    self.best_moving_vectors = self.moving_map_vec.detach().cpu().numpy()
                
                self.best_loss = eval_loss.detach().cpu().item()

            joint_ori_error, joint_pos_error, joint_state_error, cam_rotation_error, cam_translation_error, soft_iou, valid = self.compute_estimation_error(gt_joint_type, gt_joint_axis, gt_joint_pos, gt_joint_value)
            wandb.log({f"Eval/{self.loss_func} error": eval_loss.detach().cpu().item(),
                       "Eval/joint axis error": joint_ori_error,
                       "Eval/joint position error": joint_pos_error,
                       "Eval/joint state error": joint_state_error,
                       "Eval/camera rotation error": cam_rotation_error,
                       "Eval/camera translation error": cam_translation_error,
                       "Eval/moving map mIOU": soft_iou,
                       "time": end_time - start_time,
                       "lr": self.lr_scheduler.get_last_lr()[-1]}, step=i + 1)
            if self.vis_pcd and (self.current_step % 20 == 0 or self.current_step == self.steps - 1):
                self.visualize(gt_joint_axis, gt_joint_pos, vis_type="static")
                self.visualize(gt_joint_axis, gt_joint_pos, vis_type="dynamic")
            tbar.set_description("Loss: {:.6f} Joint Axis Error: {:.6f} Joint Pos Error: {:.6f} Joint State Error: {:.6f}"\
                                .format(loss.item(), joint_ori_error, joint_pos_error, joint_state_error))


    def compute_estimation_error(self, gt_joint_type: str, gt_joint_axis: np.ndarray, gt_joint_pos: np.ndarray, gt_joint_value: np.ndarray) -> Tuple[float, float, float, float, float, float, bool]:
        # joint estimation
        pred_joint_axis = self.joint_axis.detach().cpu().numpy()
        pred_joint_axis = pred_joint_axis / np.linalg.norm(pred_joint_axis)
        pred_joint_pos = self.joint_pos.detach().cpu().numpy()
        pred_joint_value = self.joint_state.detach().cpu().numpy()

        joint_ori_error = np.arccos(np.abs(np.dot(pred_joint_axis, gt_joint_axis)))
        if np.any(np.isnan(joint_ori_error)):
            print("pred_joint_ori:", pred_joint_axis)
            print("joint type:", gt_joint_type)
            return 0, 0, 0, 0, 0, 0, False

        n = np.cross(pred_joint_axis, gt_joint_axis)
        joint_pos_error = np.abs(np.dot(n, (pred_joint_pos - gt_joint_pos))) / np.linalg.norm(n)
        if np.any(np.isnan(joint_pos_error)):
            print("pred_joint_ori:", pred_joint_axis)
            print("joint type:", gt_joint_type)
            return 0, 0, 0, 0, 0, 0, False

        if gt_joint_type == "prismatic":
            joint_pos_error = 0
        
        joint_state_error = np.mean(np.abs(np.abs(gt_joint_value) - np.abs(pred_joint_value)))

        # camera estimation
        pred_camera_rotation = R.from_quat(self.camera_pose.detach().cpu().numpy()[:, :4], scalar_first=True).as_matrix()
        pred_camera_translation = self.camera_pose.detach().cpu().numpy()[:, 4:]
        rotation_error_matrix = pred_camera_rotation @ self.gt_camera_se3[:, :3, :3].transpose(0, 2, 1)
        cam_rotation_error = np.mean(np.arccos((np.trace(rotation_error_matrix, axis1=1, axis2=2) - 1) / 2))
        cam_translation_error = np.mean(np.linalg.norm(pred_camera_translation - self.gt_camera_se3[:, :3, 3], axis=1))

        # moving map
        if self.mask_type == "monst3r":
            full_moving_map = self.part_segments_full * self.moving_map_vec.reshape(1, self.part_segments_full.shape[1], 1, 1) # N, old_parts, H, W
            moving_map = torch.sum(full_moving_map, dim=1) # N, W, H
            moving_map = (moving_map - torch.amin(moving_map, dim=(1, 2), keepdim=True)) / (torch.amax(moving_map, dim=(1, 2), keepdim=True) - torch.amin(moving_map, dim=(1, 2), keepdim=True) + 1e-12)
            gt_old_moving_map = torch.logical_and(self.gt_moving_map_bool, self.old_part_segments_list).to(torch.float64)
            intersection = gt_old_moving_map * moving_map
            union = gt_old_moving_map + moving_map - intersection
            valid_union_index = torch.nonzero(torch.sum(union, dim=(1, 2)) > 1e-3, as_tuple=True)[0]
            intersection = intersection[valid_union_index, :, :]
            union = union[valid_union_index, :, :]
            soft_iou = torch.mean(torch.sum(intersection, dim=(1, 2)) / torch.sum(union, dim=(1, 2)))
            soft_iou = soft_iou.detach().cpu().numpy()
        else:
            soft_iou = 0

        return float(joint_ori_error), float(joint_pos_error), float(joint_state_error), \
            float(cam_rotation_error), float(cam_translation_error), float(soft_iou),  True
    

    def save_estimation_results(self,):
        np.save(f"{self.log_dir}/joint_axis.npy", self.best_joint_axis)
        np.save(f"{self.log_dir}/joint_pos.npy", self.best_joint_pos)
        np.save(f"{self.log_dir}/joint_value.npy", self.best_joint_state)
        np.save(f"{self.log_dir}/camera_poses.npy", self.best_camera_poses)
        if self.mask_type == "monst3r":
            full_moving_map = self.part_segments_full * \
                torch.from_numpy(self.best_moving_vectors).to(self.device).reshape(1, self.part_segments_full.shape[1], 1, 1) # N, old_parts, H, W
            moving_map = torch.sum(full_moving_map, dim=1) # N, W, H
            moving_map = (moving_map - torch.amin(moving_map, dim=(1, 2), keepdim=True)) / \
                (torch.amax(moving_map, dim=(1, 2), keepdim=True) - torch.amin(moving_map, dim=(1, 2), keepdim=True) + 1e-12)
            np.savez_compressed(f"{self.log_dir}/moving_map.npz", a=moving_map.detach().cpu().numpy())
        else:
            gt_old_moving_map = torch.logical_and(self.gt_moving_map_bool, self.old_part_segments_list).cpu().numpy()
            np.savez_compressed(f"{self.log_dir}/moving_map.npz", gt_old_moving_map)
        with open(f"{self.log_dir}/best_loss.txt", 'w') as f:
            f.write(str(self.best_loss))


    def visualize(self, gt_joint_axis: np.ndarray, gt_joint_pos: np.ndarray, vis_type: str = "static"):
        # joint
        tmp_range = np.linspace(-1, 1, 100)
        pred_joint_points = self.best_joint_pos + self.best_joint_axis * tmp_range[:, np.newaxis]
        pred_joint_rgb = np.ones_like(pred_joint_points) * np.array([[25, 25, 178]])
        pred_joint_xyzrgb = np.hstack([pred_joint_points, pred_joint_rgb])
        gt_joint_points = gt_joint_pos + gt_joint_axis * tmp_range[:, np.newaxis]
        gt_joint_rgb = np.ones_like(gt_joint_points) * np.array([[255, 0, 255]])
        gt_joint_xyzrgb = np.hstack([gt_joint_points, gt_joint_rgb])
        # pcd
        obj_pcd_list = []
        N, H, W, C = self.xyz.shape
        camera_extrinsics = torch.eye(4, dtype=torch.float64).repeat(N, 1, 1).to(self.device)
        camera_rotations = quaternion_to_matrix(torch.from_numpy(self.best_camera_poses[:, :4]).to(self.device))
        camera_extrinsics[:, :3, :3] = camera_rotations
        camera_extrinsics[:, :3, 3] = self.camera_pose[:, 4:]
        pred_transformed_xyz = torch.matmul(self.xyz.reshape(N, H*W, C), camera_extrinsics[:, :3, :3].permute(0, 2, 1)) + camera_extrinsics[:, :3, 3].reshape(N, 1, 3) # N, H*W, 3
        if vis_type == "dynamic":
            # dynamic chamfer distance
            joint_axis_norm = F.normalize(self.joint_axis.reshape(1, 3))
            if self.joint_type == "revolute":
                rot_vec = joint_axis_norm.repeat(N, 1) * self.joint_state.reshape(N, 1) # N, 3
                rotations = axis_angle_to_matrix(rot_vec) # N, 3, 3
                translations = torch.matmul((torch.eye(3).repeat(N, 1, 1).to(self.device) - rotations), self.joint_pos) # N, 3
            elif self.joint_type == "prismatic":
                rotations = torch.eye(3, dtype=torch.float64, device=self.device).repeat(N, 1, 1) # N, 3, 3
                translations = joint_axis_norm.repeat(N, 1) * self.joint_state.reshape(N, 1) # N, 3
            pred_transformed_xyz = torch.matmul(pred_transformed_xyz, rotations.permute(0, 2, 1)) + translations.reshape(N, 1, 3) # N, H*W, 3
        
        if vis_type == "static":
            gt_camera_extrinsics = torch.from_numpy(self.gt_camera_se3).to(self.device)
            gt_xyz = torch.matmul(self.xyz.reshape(N, H*W, C), gt_camera_extrinsics[:, :3, :3].permute(0, 2, 1)) + gt_camera_extrinsics[:, :3, 3].reshape(N, 1, 3) # N, H*W, 3
        if self.mask_type == "monst3r":
            full_moving_map = self.part_segments_full * \
                torch.from_numpy(self.best_moving_vectors).to(self.device).reshape(1, self.part_segments_full.shape[1], 1, 1) # N, old_parts, H, W
            moving_map = torch.sum(full_moving_map, dim=1) # N, W, H
            moving_map = (moving_map - torch.amin(moving_map, dim=(1, 2), keepdim=True)) / \
                (torch.amax(moving_map, dim=(1, 2), keepdim=True) - torch.amin(moving_map, dim=(1, 2), keepdim=True) + 1e-12)
        else:
            moving_map = torch.logical_and(self.gt_moving_map_bool, self.old_part_segments_list).to(torch.float64)
        for b in range(pred_transformed_xyz.shape[0]):
            obj_mask = self.obj_mask_bool[b].reshape(H*W)
            old_part_segment = self.old_part_segments_list[b].reshape(H*W)
            computable_mask = torch.logical_and(obj_mask, old_part_segment)
            filter_pred_transformed_xyz = pred_transformed_xyz[b, computable_mask, :].detach().cpu().numpy() # compute_mask_num, 3
            norm_moving_map_b = moving_map[b].reshape(H*W)
            filter_norm_moving_map = norm_moving_map_b[computable_mask].detach().cpu().numpy() # compute_mask_num
            
            obj_pcd_color = 255 * np.vstack([filter_norm_moving_map, 1 - filter_norm_moving_map, np.zeros_like(filter_norm_moving_map)]).T
            obj_xyzrgb = np.hstack([filter_pred_transformed_xyz, obj_pcd_color])
            sample_num = min(4096, obj_xyzrgb.shape[0])
            sample_index = np.random.choice(obj_xyzrgb.shape[0], sample_num, replace=False)
            obj_xyzrgb = obj_xyzrgb[sample_index]

            if vis_type == "static":    
                filter_gt_cam_transformed_xyz = gt_xyz[b, computable_mask, :].detach().cpu().numpy()
                filter_gt_cam_transformed_rgb = np.ones_like(filter_gt_cam_transformed_xyz) * np.array([[255, 255, 255]])
                gt_obj_xyzrgb = np.hstack([filter_gt_cam_transformed_xyz, filter_gt_cam_transformed_rgb])
                sample_num = min(4096, gt_obj_xyzrgb.shape[0])
                sample_index = np.random.choice(gt_obj_xyzrgb.shape[0], sample_num, replace=False)
                gt_obj_xyzrgb = gt_obj_xyzrgb[sample_index]
            elif vis_type == "dynamic":
                gt_static_xyz = self.surface_static_xyz.detach().cpu().numpy()
                gt_static_rgb = np.ones_like(gt_static_xyz) * np.array([[0, 255, 255]])
                gt_static_xyzrgb = np.hstack([gt_static_xyz, gt_static_rgb])
                gt_dynamic_xyz = self.surface_dynamic_xyz.detach().cpu().numpy()
                gt_dynamic_rgb = np.ones_like(gt_dynamic_xyz) * np.array([[127, 0, 255]])
                gt_dynamic_xyzrgb = np.hstack([gt_dynamic_xyz, gt_dynamic_rgb])
                gt_xyzrgb = np.vstack([gt_static_xyzrgb, gt_dynamic_xyzrgb])
                sample_surface_num = min(4096, gt_xyzrgb.shape[0])
                sample_surface_index = np.random.choice(gt_xyzrgb.shape[0], sample_surface_num, replace=False)
                gt_obj_xyzrgb = gt_xyzrgb[sample_surface_index, :]
            prediction_pcd = np.vstack([pred_joint_xyzrgb, gt_joint_xyzrgb, obj_xyzrgb, gt_obj_xyzrgb])
            wandb.log({f"Vis_{vis_type}/frame{b}": wandb.Object3D(prediction_pcd)}, self.current_step + 1)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--view_dir", type=str, required=True)
    parser.add_argument("--mask_type", type=str, choices=["gt", "monst3r"], required=True)
    parser.add_argument("--joint", type=str, choices=["revolute", "prismatic"], required=True)
    parser.add_argument("--loss", type=str, choices=["chamfer", "hausdorff"], required=True)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vis", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    joint_type_map = {"hinge": "revolute", "slider": "prismatic"}
    meta_file = "new_partnet_mobility_dataset_correct_intr_meta.json"
    with open(meta_file, "r") as f:
        data_meta = json.load(f)

    results_root_dir = "sim_data/exp_results/prediction/"
    preprocess_root_dir = "sim_data/exp_results/preprocessing/"

    view_dir = args.view_dir
    joint_data_dir = view_dir[:-7]
    opt_view = int(view_dir[-2])
    cat = joint_data_dir.split('/')[2]
    obj_id = view_dir.split('/')[3]
    mask_type = args.mask_type
    interaction_list = data_meta[cat][obj_id]["interaction_list"]
    joint_id = int(view_dir.split('/')[4][6:-3])
    interaction_dict = None
    for interaction in interaction_list:
        if joint_id == interaction["id"]:
            interaction_dict = interaction
    assert interaction_dict is not None, "does not find interaction"
    gt_joint_axis = interaction_dict["joint"]["axis"]["direction"]
    gt_joint_pos = interaction_dict["joint"]["axis"]["origin"]
    gt_joint_type = joint_type_map[interaction_dict["type"]]
    sample_rgb_index = [int(file_name[:-4]) for file_name in os.listdir(f"{joint_data_dir}/view_{opt_view}/sample_rgb/")]
    sample_rgb_index.sort()
    gt_joint_value = np.load(f"{joint_data_dir}/gt_joint_value.npy")
    sample_gt_joint_value = gt_joint_value[sample_rgb_index]

    opt_steps = args.steps
    device = torch.device(args.device)
    loss_func = args.loss
    prediction_dir = f"{results_root_dir}/{cat}/{obj_id}/joint_{joint_id}_bg/view_{opt_view}/"
    preprocess_dir = f"{preprocess_root_dir}/{cat}/{obj_id}/joint_{joint_id}_bg/view_{opt_view}/"
    log_dir = f"{results_root_dir}/{cat}/{obj_id}/joint_{joint_id}_bg/view_{opt_view}/{args.exp_name}/{mask_type}/"
    os.makedirs(f"{log_dir}/{loss_func}/{args.seed}/{args.joint}/", exist_ok=True)

    wandb.login(key=os.environ["WANDB_API_KEY"])
    run_config = {
            "mask_type": mask_type,
            "loss_function": loss_func,
            "epochs": opt_steps,
            "learning_rate": args.lr,
            "seed": args.seed,
        }

    with wandb.init(project=f"video_articulation_{args.exp_name}_{mask_type}", \
                    config=run_config, dir=f"{log_dir}/{loss_func}/{args.seed}/{args.joint}", \
                    name=f"{view_dir}/{args.joint}/seed{args.seed}/"):
        ba = BundleAdjustment(joint_data_dir, preprocess_dir, prediction_dir, opt_view, mask_type, args.joint, 
                              args.lr, loss_func, opt_steps, log_dir, device, args.seed, args.vis)
        wandb.watch(ba, log="all", log_freq=1)
        if not ba.valid:
            print("predict error")
        ba.dump_configuration()
        ba.optimize_adam(gt_joint_type, gt_joint_axis, gt_joint_pos, sample_gt_joint_value)
        ba.save_estimation_results()