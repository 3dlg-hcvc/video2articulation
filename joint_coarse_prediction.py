import numpy as np
from scipy.spatial.transform import Rotation as R
from kornia.feature import LoFTR
import cv2
import torch
import argparse
import os
from tqdm import tqdm
from typing import Tuple, List, Dict

from utils import set_seed, estimate_se3_transformation
from data import DataLoader, SimDataLoader, RealDataLoader


class CoarsePrediction():
    def __init__(self, data_loader: DataLoader, prediction_dir: str, mask_type: str, device: torch.device, seed: int = 0):
        self.data_loader = data_loader
        self.mask_type = mask_type
        self.device = device

        self.matcher = LoFTR(pretrained="indoor").to(device)

        self.H, self.W = data_loader.H, data_loader.W
        self.rgb_list, self.xyz_list =  self.data_loader.load_rgbd_video()
        if isinstance(data_loader, SimDataLoader):
            self.gt_camera_se3 = self.data_loader.load_gt_camera_pose_se3()
        self.camera2label = [self.data_loader.load_gt_init_camera_pose_se3()]

        if isinstance(data_loader, RealDataLoader):
            self.obj_mask_list = self.data_loader.load_obj_mask()
        else:
            self.obj_mask_list = None
        if isinstance(data_loader, SimDataLoader) and mask_type == "gt":
            self.dynamic_mask_list = self.data_loader.load_gt_moving_map()
        else:
            self.dynamic_mask_list = self.data_loader.load_monst3r_moving_map()
        self.prediction_dir = prediction_dir
        self.prediction_joint_metrics = None
        self.prediction_joint_type = None
        self.seed = seed
        set_seed(seed)


    def filter_match(self, kp1: np.ndarray, kp2: np.ndarray, thresh: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        dist = np.linalg.norm((kp1 - kp2), axis=1)
        kp1 = kp1[dist < thresh]
        kp2 = kp2[dist < thresh]
        return kp1, kp2


    def compute_match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # img1_raw = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img1_raw = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img1_torch = torch.Tensor(img1_raw).cuda() / 255.
        img1_torch = torch.reshape(img1_torch, (1, 1, img1_torch.shape[0], img1_torch.shape[1]))
        # img2_raw = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        img2_raw = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        img2_torch = torch.Tensor(img2_raw).cuda() / 255.
        img2_torch = torch.reshape(img2_torch, (1, 1, img2_torch.shape[0], img2_torch.shape[1]))

        input = {"image0": img1_torch, "image1": img2_torch}
        correspondences_dict = self.matcher(input)

        mkpts0 = correspondences_dict['keypoints0'].cpu().numpy()
        mkpts1 = correspondences_dict['keypoints1'].cpu().numpy()
        mconf = correspondences_dict['confidence'].cpu().numpy()
        return mkpts0, mkpts1, mconf


    def align_pc(self, curr_pc: np.ndarray, base_kp: np.ndarray, curr_kp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H = curr_pc.shape[0]
        W = curr_pc.shape[1]
        curr2base = estimate_se3_transformation(base_kp, curr_kp)
        rotation = curr2base[:3, :3]
        translation = curr2base[:3, 3]
        align_curr_pc = (rotation @ curr_pc.reshape(-1, 3).T).T + translation.reshape(1, 3)
        return align_curr_pc.reshape(H, W, 3), curr2base


    def align_view(self,) -> List[np.ndarray]:
        pc_list = []
        camera2label = self.camera2label[0].copy()
        for i in range(len(self.rgb_list) - 1):
            base_pc = self.xyz_list[i]
            next_pc = self.xyz_list[i + 1]

            mkpts0, mkpts1, conf = self.compute_match(self.rgb_list[i], self.rgb_list[i + 1])
            match_mask = conf > 0.95
            mkpts0 = mkpts0[match_mask].astype(np.uint32)
            mkpts1 = mkpts1[match_mask].astype(np.uint32)

            dynamic_mask = self.dynamic_mask_list[i]
            if self.obj_mask_list is not None:
                obj_mask = self.obj_mask_list[i]
                static_mask = ((~dynamic_mask) & obj_mask)
                static_mask = ~dynamic_mask
            else:
                static_mask = ~dynamic_mask
            static_index = np.nonzero(static_mask[mkpts0[:, 1], mkpts0[:, 0]])[0]
            static_pts0 = mkpts0[static_index]
            static_pts1 = mkpts1[static_index]

            static_kp0 = base_pc[static_pts0[:, 1], static_pts0[:, 0]]
            static_kp1 = next_pc[static_pts1[:, 1], static_pts1[:, 0]]
            static_kp0, static_kp1 = self.filter_match(static_kp0, static_kp1)
            aligned_next_pc, next2base = self.align_pc(next_pc, static_kp0, static_kp1)

            base_pc_origin = (camera2label[:3, :3] @ base_pc.reshape(-1, 3).T).T + camera2label[:3, 3].reshape(1, 3)
            base_pc_origin = np.reshape(base_pc_origin, (self.H, self.W, 3))
            next_pc_origin = (camera2label[:3, :3] @ aligned_next_pc.reshape(-1, 3).T).T + camera2label[:3, 3].reshape(1, 3)
            next_pc_origin = np.reshape(next_pc_origin, (self.H, self.W, 3))

            camera2label = np.dot(camera2label, next2base)
            self.camera2label.append(camera2label)
            if i == 0:
                pc_list.append(base_pc_origin)
            pc_list.append(next_pc_origin)
        return pc_list
    

    def estimate_joint_transformation(self, base_kp: np.ndarray, curr_kp: np.ndarray, type: str, RANSAC: bool) -> Tuple[np.ndarray, np.ndarray]:
        curr2base = None
        inlier = None
        if RANSAC:
            k = 50
            inlier_thresh = 1e-2
            d = int(base_kp.shape[0] * 0.4)
            best_se3 = None
            best_error = 10000
            best_inlier = None
            inlier_list = []
            se3_list = []
            inlier_index_list = []
            for _ in range(k):
                init_sample = np.random.choice(base_kp.shape[0], 10, replace=False)
                init_kp1 = base_kp[init_sample]
                init_kp2 = curr_kp[init_sample]
                
                if type == "revolute":
                    se3 = estimate_se3_transformation(init_kp1, init_kp2)
                elif type == "prismatic":
                    only_translation = np.mean((init_kp1 - init_kp2), axis=0)
                    se3 = np.eye(4)
                    se3[:3, 3] = only_translation
                se3_list.append(se3)
                rotation = se3[:3, :3]
                translation = se3[:3, 3]

                transform_kp2 = curr_kp @ rotation.T + translation
                dist = np.linalg.norm((base_kp - transform_kp2), axis=1)
                inlier = np.nonzero(dist < inlier_thresh)[0]
                inlier_list.append(inlier.shape[0])
                inlier_index_list.append(inlier)
                if inlier.shape[0] > d:
                    if type == "revolute":
                        se3 = estimate_se3_transformation(base_kp[inlier], curr_kp[inlier])
                    elif type == "prismatic":
                        only_translation = np.mean((base_kp[inlier] - curr_kp[inlier]), axis=0)
                        se3 = np.eye(4)
                        se3[:3, 3] = only_translation
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
            if type == "revolute":
                curr2base = estimate_se3_transformation(base_kp, curr_kp)
            elif type == "prismatic":
                only_translation = np.mean((base_kp - curr_kp), axis=0)
                curr2base = np.eye(4)
                curr2base[:3, 3] = only_translation
            inlier = np.arange(base_kp.shape[0])
        return curr2base, inlier


    def estimate_joint_single(self, base_kp: np.ndarray, curr_kp: np.ndarray, RANSAC: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
        curr2base, revolute_inlier = self.estimate_joint_transformation(base_kp, curr_kp, "revolute", RANSAC)
        rotation = curr2base[:3, :3]
        translation = curr2base[:3, 3]

        result = {}
        joint_rotvec = R.from_matrix(rotation.T).as_rotvec()
        revolute_joint_axis = joint_rotvec / np.linalg.norm(joint_rotvec)
        det = np.linalg.det(np.eye(3) - rotation)
        valid = True
        try:
            revolute_joint_pos = np.linalg.inv(np.eye(3) - rotation) @ translation
            revolute_joint_pos = revolute_joint_pos - np.dot(revolute_joint_pos, revolute_joint_axis) * revolute_joint_axis
        except:
            det = 0
            revolute_joint_pos = np.zeros(3)
            valid = False
        if abs(det) < 1e-17:
            print("angle too small!")
            valid = False
        rotate_curr_kp = (curr_kp[revolute_inlier] - revolute_joint_pos) @ rotation.T + revolute_joint_pos
        rotation_error = np.mean((base_kp[revolute_inlier] - rotate_curr_kp) ** 2)
        result["revolute"] = {"X": curr_kp[revolute_inlier], "Y": base_kp[revolute_inlier], "axis": revolute_joint_axis, "pos": revolute_joint_pos, "error": rotation_error, "det": det,  "valid": valid}

        only_translation_se3, prismatic_inlier = self.estimate_joint_transformation(base_kp, curr_kp, "prismatic", RANSAC)
        only_translation = only_translation_se3[:3, 3]
        prismatic_joint_axis = only_translation / np.linalg.norm(only_translation)
        prismatic_joint_pos = base_kp[0]
        translate_curr_kp = curr_kp[prismatic_inlier] + only_translation
        translation_error = np.mean((base_kp[prismatic_inlier] - translate_curr_kp) ** 2)
        result["prismatic"] = {"X": curr_kp[prismatic_inlier], "Y": base_kp[prismatic_inlier], "axis": prismatic_joint_axis, "pos": prismatic_joint_pos, "error": translation_error, "valid": True}

        return result
    

    def estimate_joint_all(self, result_list: List[Dict[str, Dict[str, np.ndarray]]]) -> Tuple[Dict[str, Dict[str, np.ndarray]], str]:
        revolute_error = 0
        prismatic_error = 0
        revolute_joint_axis = 0
        prismatic_joint_axis = 0
        revolute_joint_pos = 0
        revolute_count = 0
        revolute_max_det = -1
        revolute_max_det_index = -1
        joint_type_vote = 0
        for index, result in enumerate(result_list):
            if result["revolute"]["valid"]:
                revolute_error += result["revolute"]["error"]
                revolute_joint_axis += result["revolute"]["axis"]
                revolute_joint_pos += result["revolute"]["pos"]
                revolute_count += 1
            if result["revolute"]["det"] > revolute_max_det:
                revolute_max_det_index = index
                revolute_max_det = result["revolute"]["det"]

            prismatic_error += result["prismatic"]["error"]
            prismatic_joint_axis += result["prismatic"]["axis"]
            if result["revolute"]["valid"]:
                if result["revolute"]["error"] < result["prismatic"]["error"]:
                    joint_type_vote += 1
                else:
                    joint_type_vote -= 1
        if joint_type_vote > 0:
            pred_joint_type = "revolute"
        elif joint_type_vote < 0:
            pred_joint_type = "prismatic"
        else:
            if revolute_count == 0:
                revolute_error = result_list[revolute_max_det_index]["revolute"]["error"]
            else:
                revolute_error = revolute_error / revolute_count
            prismatic_error = prismatic_error / len(result_list)
            pred_joint_type = "revolute" if revolute_error < prismatic_error else "prismatic"
        if revolute_count == 0:
            revolute_joint_axis = result_list[revolute_max_det_index]["revolute"]["axis"]
            revolute_joint_pos = result_list[revolute_max_det_index]["revolute"]["pos"]
        else:
            revolute_joint_axis = revolute_joint_axis / np.linalg.norm(revolute_joint_axis)
            revolute_joint_pos = revolute_joint_pos / revolute_count
        prismatic_joint_axis = prismatic_joint_axis / np.linalg.norm(prismatic_joint_axis)
        prismatic_joint_pos = np.zeros(3)
        pred_joint_metrics = {"revolute": {"axis": revolute_joint_axis, "pos": revolute_joint_pos}, 
                              "prismatic": {"axis": prismatic_joint_axis, "pos": prismatic_joint_pos},}
        
        return pred_joint_metrics, pred_joint_type


    def compute_average_rotation_angle(self, X: np.ndarray, Y: np.ndarray, joint_axis: np.ndarray, joint_pos: np.ndarray) -> np.ndarray:
        # Normalize the rotation axis
        joint_axis = joint_axis / np.linalg.norm(joint_axis)
        
        # Initialize sums for sine and cosine components
        sin_sum = 0
        cos_sum = 0
        
        for x, y in zip(X, Y):
            x = x - joint_pos
            y = y - joint_pos
            # Project points onto the plane perpendicular to the axis
            x_perp = x - np.dot(x, joint_axis) * joint_axis
            y_perp = y - np.dot(y, joint_axis) * joint_axis
            
            # Normalize the projected vectors
            x_perp = x_perp / np.linalg.norm(x_perp)
            y_perp = y_perp / np.linalg.norm(y_perp)
            
            # Compute cosine and sine for this pair
            cos_theta = np.dot(x_perp, y_perp)
            sin_theta = np.dot(joint_axis, np.cross(x_perp, y_perp))
            
            # Accumulate sine and cosine
            cos_sum += cos_theta
            sin_sum += sin_theta
        
        # Compute the average angle
        angle = np.arctan2(sin_sum, cos_sum)
        return angle


    def compute_average_translation_distance(self, X: np.ndarray, Y: np.ndarray, joint_axis: np.ndarray) -> np.ndarray:
        joint_axis = joint_axis / np.linalg.norm(joint_axis)
        translation_vec = Y - X
        translation_dist = np.mean(np.dot(translation_vec, joint_axis))
        return translation_dist


    def estimate_joint(self,) -> Tuple[Dict[str, Dict[str, np.ndarray]], str]:
        # img_list = os.listdir(self.sample_rgb_dir)
        # img_list.sort()
        pc_list = self.align_view()
        result_list = []
        pair_list = []
        for interval in [1, 2, 3]:
            for i in range(0, len(self.rgb_list) - interval, 1):
                mkpts0, mkpts1, conf = self.compute_match(self.rgb_list[i], self.rgb_list[i + interval])
                match_mask = conf > 0.9
                mkpts0 = mkpts0[match_mask].astype(np.uint32)
                mkpts1 = mkpts1[match_mask].astype(np.uint32)

                dynamic_mask = self.dynamic_mask_list[i]
                if self.obj_mask_list is not None:
                    obj_mask = self.obj_mask_list[i]
                    dynamic_mask = dynamic_mask & obj_mask
                dynamic_index = np.nonzero(dynamic_mask[mkpts0[:, 1], mkpts0[:, 0]])[0]
                dynamic_pts0 = mkpts0[dynamic_index]
                dynamic_pts1 = mkpts1[dynamic_index]

                base_pc_origin = pc_list[i]
                next_pc_origin = pc_list[i + interval]

                dynamic_kp0 = base_pc_origin[dynamic_pts0[:, 1], dynamic_pts0[:, 0]]
                dynamic_kp1 = next_pc_origin[dynamic_pts1[:, 1], dynamic_pts1[:, 0]]
                dynamic_kp0, dynamic_kp1 = self.filter_match(dynamic_kp0, dynamic_kp1)

                if dynamic_kp0.shape[0] > 80:
                    result_i = self.estimate_joint_single(dynamic_kp0, dynamic_kp1, RANSAC=True)
                    result_list.append(result_i)
                    pair_list.append((i, i + interval))
        if len(result_list) > 0:
            pred_joint_metrics, pred_joint_type = self.estimate_joint_all(result_list)
            for joint_type in pred_joint_metrics.keys():
                joint_value_per_frame = 0
                for pair, result in zip(pair_list, result_list):
                    if joint_type == "revolute":
                        angle = self.compute_average_rotation_angle(result[joint_type]["X"], result[joint_type]["Y"], pred_joint_metrics[joint_type]["axis"], pred_joint_metrics[joint_type]["pos"])
                        angle_per_frame = angle / (pair[1] - pair[0])
                        joint_value_per_frame += angle_per_frame
                    elif joint_type == "prismatic":
                        distance = self.compute_average_translation_distance(result[joint_type]["X"], result[joint_type]["Y"], pred_joint_metrics[joint_type]["axis"])
                        distance_per_frame = distance / (pair[1] - pair[0])
                        joint_value_per_frame += distance_per_frame
                joint_value_per_frame = joint_value_per_frame / (len(result_list))
                pred_joint_metrics[joint_type]["average_value"] = joint_value_per_frame
        else:
            pred_joint_metrics = None
            pred_joint_type = None
        self.prediction_joint_metrics = pred_joint_metrics
        self.prediction_joint_type = pred_joint_type
        return pred_joint_metrics, pred_joint_type


    def save_prediction_results(self,):
        if self.prediction_joint_metrics is not None:
            prediction_result_dir = f"{self.prediction_dir}/coarse_prediction/{self.mask_type}/{self.seed}/"
            os.makedirs(prediction_result_dir, exist_ok=True)
            os.makedirs(f"{prediction_result_dir}/revolute/", exist_ok=True)
            os.makedirs(f"{prediction_result_dir}/prismatic/", exist_ok=True)
            np.save(f"{prediction_result_dir}/revolute/joint_axis.npy", self.prediction_joint_metrics["revolute"]["axis"])
            np.save(f"{prediction_result_dir}/revolute/joint_pos.npy", self.prediction_joint_metrics["revolute"]["pos"])
            revolute_joint_value_list = np.arange(len(self.rgb_list)) * self.prediction_joint_metrics["revolute"]["average_value"]
            np.save(f"{prediction_result_dir}/revolute/joint_value.npy", revolute_joint_value_list)
            np.save(f"{prediction_result_dir}/prismatic/joint_axis.npy", self.prediction_joint_metrics["prismatic"]["axis"])
            np.save(f"{prediction_result_dir}/prismatic/joint_pos.npy", self.prediction_joint_metrics["prismatic"]["pos"])
            prismatic_joint_value_list = np.arange(len(self.rgb_list)) * self.prediction_joint_metrics["prismatic"]["average_value"]
            np.save(f"{prediction_result_dir}/prismatic/joint_value.npy", prismatic_joint_value_list)
            os.makedirs(f"{prediction_result_dir}/cam_pose/", exist_ok=True)
            for frame, cam2label in enumerate(self.camera2label):
                np.save(f"{prediction_result_dir}/cam_pose/cam2label_{frame}.npy", cam2label)


    def compute_estimation_error(self, gt_joint_type: str, gt_joint_axis: np.ndarray, gt_joint_pos: np.ndarray, gt_joint_value: np.ndarray) -> Tuple[float, float, float, bool, float, float]:
        if self.prediction_joint_metrics is None or self.prediction_joint_metrics[self.prediction_joint_type] is None:
            return np.pi / 2, 1, np.pi / 2 if gt_joint_type == "revolute" else 1, True, 1, np.pi / 2
        
        joint_type_error = (self.prediction_joint_type != gt_joint_type)
        pred_joint_axis = self.prediction_joint_metrics[self.prediction_joint_type]["axis"]
        pred_joint_pos = self.prediction_joint_metrics[self.prediction_joint_type]["pos"]

        joint_ori_error = np.arccos(np.abs(np.dot(pred_joint_axis, gt_joint_axis)))
        if np.any(np.isnan(joint_ori_error)):
            print("pred_joint_ori:", pred_joint_axis)
            print("joint type:", gt_joint_type)
            joint_ori_error = np.pi / 2

        n = np.cross(pred_joint_axis, gt_joint_axis)
        joint_pos_error = np.abs(np.dot(n, (pred_joint_pos - gt_joint_pos))) / np.linalg.norm(n)
        if np.any(np.isnan(joint_pos_error)):
            joint_pos_error = 1
        if gt_joint_type == "prismatic":
            joint_pos_error = 0

        if self.prediction_joint_type == "revolute":
            pred_joint_value = np.arange(len(self.img_list)) * self.prediction_joint_metrics["revolute"]["average_value"]
        else:
            pred_joint_value = np.arange(len(self.img_list)) * self.prediction_joint_metrics["prismatic"]["average_value"]
        joint_state_error = np.mean(np.abs(np.abs(gt_joint_value) - np.abs(pred_joint_value)))
        if np.any(np.isnan(joint_state_error)):
            joint_state_error = 1
        
        # camera estimation
        pred_camera_se4 = np.stack(self.camera2label)
        pred_camera_rotation = pred_camera_se4[:, :3, :3]
        pred_camera_translation = pred_camera_se4[:, :3, 3]
        rotation_error_matrix = pred_camera_rotation @ self.gt_camera_se3[:, :3, :3].transpose(0, 2, 1)
        cam_rotation_error = np.mean(np.arccos((np.trace(rotation_error_matrix, axis1=1, axis2=2) - 1) / 2))
        if np.any(np.isnan(cam_rotation_error)):
            cam_rotation_error = 1
        cam_translation_error = np.mean(np.linalg.norm(pred_camera_translation - self.gt_camera_se3[:, :3, 3], axis=1))
        if np.any(np.isnan(cam_translation_error)):
            cam_translation_error = 1

        return float(joint_ori_error), float(joint_pos_error), float(joint_state_error), joint_type_error, cam_translation_error, cam_rotation_error
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, choices=["sim", "real"], required=True)
    parser.add_argument("--view_dir", type=str, required=True)
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument("--prediction_dir", type=str, required=True)
    parser.add_argument("--meta_file_path", type=str, default="new_partnet_mobility_dataset_correct_intr_meta.json", help="Only required for sim data, the path to the metadata file.")
    parser.add_argument("--mask_type", type=str, choices=["gt", "monst3r"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda")
    if args.data_type == "sim":
        data_loader = SimDataLoader(args.view_dir, args.preprocess_dir, args.meta_file_path, None)
    elif args.data_type == "real":
        data_loader = RealDataLoader(args.view_dir, args.preprocess_dir)
    else:
        raise ValueError("Unsupported data type. Choose 'sim' or 'real'.")
    # data_loader = SimDataLoader(args.meta_file_path, args.view_dir, args.preprocess_dir, None)
    coarse_predictor = CoarsePrediction(data_loader, args.prediction_dir, args.mask_type, device, args.seed)
    coarse_predictor.estimate_joint()
    coarse_predictor.save_prediction_results()
