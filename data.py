import numpy as np
import cv2
from PIL import Image
import trimesh
from scipy.spatial.transform import Rotation as R
from yourdfpy import URDF
import pickle
import json
import os
import sys
import glob
from pathlib import Path
from typing import Tuple, List, Dict

from utils import find_movable_part, precompute_object2label, depth2xyz, precompute_camera2label
from data_utils import descendant_links, load_joint_cfg, sample_urdf_pcd, remove_overlay


class SimDataLoader:
    def __init__(self, meta_file_path: str, view_dir: str, preprocess_dir: str, partnet_mobility_dir: str = "partnet-mobility-v0"):
        if meta_file_path is not None:
            self.joint_type_map = {"hinge": "revolute", "slider": "prismatic"}
            with open(meta_file_path, "r") as f:
                self.data_meta = json.load(f)
        self.W = 640
        self.H = 480

        norm_view_dir = os.path.normpath(view_dir)
        self.view_dir = norm_view_dir
        self.joint_data_dir = os.path.dirname(norm_view_dir)
        print(self.joint_data_dir)
        folder_names = norm_view_dir.strip("/").split("/")
        self.opt_view = int(folder_names[-1][-1])
        self.joint_id = int(folder_names[-2][-4])
        self.obj_id = folder_names[-3]
        self.cat = folder_names[-4]

        self.surface_dir = f"{self.joint_data_dir}/view_init"
        
        actor_pose_path = f"{self.joint_data_dir}/actor_pose.pkl"
        with open(actor_pose_path, 'rb') as f:
            obj_pose_dict = pickle.load(f)
        self.actor_list = []
        for actor_name in obj_pose_dict.keys():
            self.actor_list.append(int(actor_name[6:]))
        self.moving_part_id = find_movable_part(obj_pose_dict)
        init_base_pose = obj_pose_dict["actor_6"][0]
        self.object2label = precompute_object2label(init_base_pose)

        self.intrinsics = np.load(f"{norm_view_dir}/intrinsics.npy")
        self.sample_rgb_dir = f"{norm_view_dir}/sample_rgb"
        self.segment_dir = f"{norm_view_dir}/segment"
        img_list = os.listdir(self.sample_rgb_dir)
        img_list.sort()
        self.sample_rgb_index = [int(file_name[:-4]) for file_name in os.listdir(self.sample_rgb_dir)]
        self.sample_rgb_index.sort()

        if preprocess_dir is not None:
            self.monst3r_dir = f"{preprocess_dir}/monst3r"
            self.part_segment_dir = f"{preprocess_dir}/video_segment_reverse"

        if partnet_mobility_dir is not None:
            self.urdf_path = Path(f"{partnet_mobility_dir}/{self.obj_id}/mobility.urdf").expanduser().resolve()
            self.urdf_dir = self.urdf_path.parent


    def load_obj_surface(self, sample_num: int = 480 * 640, return_segments: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray]:
        surface_img = []
        surface_xyz = []
        surface_mask = []
        surface_dynamic_mask_list = []
        num_frames = len(os.listdir("{}/rgb".format(self.surface_dir)))
        for i in range(num_frames):
            img = cv2.imread("{}/rgb/{}".format(self.surface_dir, "%06d.png" % i))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            surface_img.append(img)
            xyz = np.load("{}/xyz/{}".format(self.surface_dir, "%06d.npz" % i))['a']
            surface_xyz.append(xyz)
            segment = np.load("{}/segment/{}".format(self.surface_dir, "%06d.npz" % i))['a']
            mask = segment == -1
            surface_dynamic_mask = segment == self.moving_part_id
            for actor_id in self.actor_list:
                mask_id = segment == actor_id
                mask = np.logical_or(mask, mask_id)
            surface_mask.append(mask)
            surface_dynamic_mask_list.append(surface_dynamic_mask)
        surface_rgb = np.stack(surface_img).reshape(-1, 3)
        surface_xyz = np.stack(surface_xyz).reshape(-1, 3)
        surface_segment = np.stack(surface_mask).flatten()

        sample_index = np.random.choice(np.arange(surface_rgb.shape[0]), sample_num, replace=False)
        surface_rgb = surface_rgb[sample_index]
        surface_xyz = surface_xyz[sample_index]
        surface_segment = surface_segment[sample_index]
        surface_xyz = np.dot(surface_xyz, self.object2label[:3, :3].T) + self.object2label[:3, 3]

        if return_segments:
            surface_dynamic_segment = np.stack(surface_dynamic_mask_list).flatten()
            surface_dynamic_segment = surface_dynamic_segment[sample_index]
            surface_static_segment = np.logical_and(surface_segment, ~surface_dynamic_segment)
            surface_static_xyz = surface_xyz[surface_static_segment]
            surface_dynamic_xyz = surface_xyz[surface_dynamic_segment]
            surface_xyz = surface_xyz[surface_segment]
            surface_rgb = surface_rgb[surface_segment]
            # surface_static_xyz = np.dot(surface_static_xyz, self.object2label[:3, :3].T) + self.object2label[:3, 3]
            # surface_dynamic_xyz = np.dot(surface_dynamic_xyz, self.object2label[:3, :3].T) + self.object2label[:3, 3]
            return surface_rgb, surface_xyz, surface_static_xyz, surface_dynamic_xyz
        else:
            surface_xyz = surface_xyz[surface_segment]
            surface_rgb = surface_rgb[surface_segment]
            return surface_rgb, surface_xyz


    def load_rgbd_video(self,) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        depth_list = glob.glob(f"{self.view_dir}/depth/*.npz")
        depth_list.sort()
        xyz = []
        rgb_list = glob.glob(f"{self.view_dir}/rgb/*.jpg")
        rgb_list.sort()
        rgb = []
        for i in self.sample_rgb_index:
            depth = np.load(depth_list[i])['a']
            xyz.append(depth2xyz(depth, self.intrinsics)) # in opengl coordinate
            bgr = cv2.imread(rgb_list[i])
            rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return rgb, xyz


    def load_monst3r_moving_map(self,) -> List[np.ndarray]:
        monst3r_mask_list = []
        for i in range(len(self.sample_rgb_index)):
            pred_mask_img = Image.open(f"{self.monst3r_dir}/dynamic_mask_{i}.png").convert('L')
            pred_mask_img = pred_mask_img.resize((self.W, self.H), Image.BICUBIC)
            pred_mask = np.array(pred_mask_img)
            pred_mask = (pred_mask / 255.).astype(np.bool_)
            monst3r_mask_list.append(pred_mask)
        return monst3r_mask_list
    

    def load_part_segmentation(self, obj_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        segments_map_list = glob.glob(f"{self.part_segment_dir}/small/final-output/*.npz")
        segments_map_list.sort(key=lambda x: int(x[x.rfind('_') + 1:-4]), reverse=True)
        segments_map_list = [segments_map_list[i] for i in self.sample_rgb_index]
        origin_segments_list = []
        for segments_map in segments_map_list:
            origin_segments_list.append(np.load(segments_map)['a'][:, 0, :, :])
        non_overlap_segments_list = remove_overlay(origin_segments_list)

        initial_segments = non_overlap_segments_list[0]
        old_segment_id_list = []
        initial_obj_mask = obj_mask[0]
        part_segments_list = []
        old_part_segments_list = []
        for segment_id in range(initial_segments.shape[0]):
            part_segment0 = initial_segments[segment_id]
            part_occupation = np.logical_and(part_segment0, initial_obj_mask)
            if np.sum(part_occupation) > 100:
                old_segment_id_list.append(segment_id)

        for frame_id, segments_map in enumerate(segments_map_list):
            part_segments = non_overlap_segments_list[frame_id]
            obj_segments = obj_mask[frame_id] # H, W
            part_segments = np.logical_and(part_segments[old_segment_id_list], obj_segments[np.newaxis, ...])
            part_segments_list.append(part_segments) # old_parts, H, W
            old_part_segments = np.zeros_like(obj_segments, dtype=np.bool_)
            for part_id in range(part_segments.shape[0]):
                old_part_segments = np.logical_or(old_part_segments, part_segments[part_id])
            old_part_segments_list.append(old_part_segments) # H, W

        old_part_segments_list = np.stack(old_part_segments_list) # N, H, W
        part_segments_list = np.stack(part_segments_list) # N, old_parts, H, W
        return part_segments_list, old_part_segments_list


    def load_gt_camera_pose_se3(self,) -> np.ndarray:
        gt_camera_pose = np.load(f"{self.view_dir}/camera_pose.npy")
        gt_camera_se3 = []
        for i in self.sample_rgb_index:
            gt_camera_se3_i= precompute_camera2label(gt_camera_pose[i], self.object2label)
            gt_camera_se3.append(gt_camera_se3_i)
        gt_camera_se3 = np.stack(gt_camera_se3) # N, 4, 4
        return gt_camera_se3
    

    def load_gt_moving_map(self,) -> List[np.ndarray]:
        gt_moving_map = []
        for i in range(len(self.sample_rgb_index)):
            segment = np.load(f"{self.segment_dir}/{self.sample_rgb_index[i]:06d}.npz")['a']
            dynamic_mask = segment == self.moving_part_id
            gt_moving_map.append(dynamic_mask)
        return gt_moving_map


    def load_obj_mask(self,) -> np.ndarray:
        obj_mask_list = []
        for i in range(len(self.sample_rgb_index)):
            segment = np.load(f"{self.segment_dir}/{self.sample_rgb_index[i]:06d}.npz")['a']
            obj_mask = segment == -1
            for actor_id in self.actor_list:
                obj_mask_id = segment == actor_id
                obj_mask = np.logical_or(obj_mask, obj_mask_id)
            obj_mask_list.append(obj_mask)
        return np.stack(obj_mask_list)
    

    def load_gt_pcd(self,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        print(f"Loading URDF from {self.urdf_path}")
        robot = URDF.load(self.urdf_path, mesh_dir=self.urdf_dir)

        joint_name = f"joint_{self.joint_id}"
        moving_links = descendant_links(robot, joint_name)
        if not moving_links:
            print(f"[warn] no children found for joint '{joint_name}'")
            return

        # Build and apply joint configuration.
        joint_name_list = f"{self.joint_data_dir}/joint_id_list.txt"
        joint_value_list = f"{self.joint_data_dir}/qpos.npy"
        cfg = load_joint_cfg(Path(joint_name_list), Path(joint_value_list))
        unknown = [j for j in cfg if j not in robot.joint_map]
        if unknown:
            sys.exit(f"Unknown joints in provided list: {', '.join(unknown)}")

        robot.update_cfg(cfg)  # ← sets internal configuration used by get_transform

        print("Sampling full point clouds …")
        link_map = robot.link_map
        full_link_list = link_map.keys()
        gt_full_pcd = sample_urdf_pcd(self.urdf_dir, full_link_list, robot, final_pts_num=10000)

        print("Sampling point clouds for moving links …")
        gt_moving_pcd = sample_urdf_pcd(self.urdf_dir, moving_links, robot, final_pts_num=10000)

        print("Sampling point clouds for static links …")
        static_links = [link for link in full_link_list if link not in moving_links]
        gt_static_pcd = sample_urdf_pcd(self.urdf_dir, static_links, robot, final_pts_num=10000)

        return gt_full_pcd, gt_moving_pcd, gt_static_pcd
    
    
    def load_gt_joint_params(self,) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        interaction_list = self.data_meta[self.cat][self.obj_id]["interaction_list"]
        interaction_dict = None
        for interaction in interaction_list:
            if self.joint_id == interaction["id"]:
                interaction_dict = interaction
        assert interaction_dict is not None, "does not find interaction"
        gt_joint_axis = interaction_dict["joint"]["axis"]["direction"]
        gt_joint_pos = interaction_dict["joint"]["axis"]["origin"]
        gt_joint_type = self.joint_type_map[interaction_dict["type"]]
        gt_joint_value = np.load(f"{self.joint_data_dir}/gt_joint_value.npy")
        sample_gt_joint_value = gt_joint_value[self.sample_rgb_index]
        return gt_joint_type, np.array(gt_joint_axis), np.array(gt_joint_pos), sample_gt_joint_value