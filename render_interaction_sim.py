import sapien.core as sapien
import numpy as np
from PIL import Image
import time
import os
import moviepy
import moviepy.video.io.ImageSequenceClip
from queue import Queue
import json
import pickle
import shutil
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import argparse
from typing import Tuple, List, Dict


def pics2video(frames_dir: str, video_dst: str, fps: float = 15):
    frames_name = sorted(os.listdir(frames_dir))
    frames_path = [frames_dir+frame_name for frame_name in frames_name]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(frames_path, fps=fps)
    clip.write_videofile(video_dst, codec='libx264')


def get_joints_dict(articulation: sapien.Articulation) -> Tuple[Dict[str, sapien.Joint], List[str]]:
    joints = articulation.get_active_joints()
    joint_names =  [joint.name for joint in joints]
    assert len(joint_names) == len(set(joint_names)), 'Joint names are assumed to be unique.'
    return {joint.name: joint for joint in joints}, joint_names


def get_rgba_img(camera: sapien.CameraEntity) -> np.ndarray:
    # ---------------------------------------------------------------------------- #
    # RGBA
    # ---------------------------------------------------------------------------- #
    rgba = camera.get_float_texture('Color')  # [H, W, 4]
    # An alias is also provided
    # rgba = camera.get_color_rgba()  # [H, W, 4]
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")

    return rgba_img


def get_xyz_img(camera: sapien.CameraEntity, coords: str = "world") -> np.ndarray:
    # ---------------------------------------------------------------------------- #
    # XYZ position in the camera space
    # ---------------------------------------------------------------------------- #
    # Each pixel is (x, y, z, render_depth) in camera space (OpenGL/Blender)
    position = camera.get_float_texture('Position')  # [H, W, 4]

    # OpenGL/Blender: y up and -z forward
    # points_opengl = position[..., :3][position[..., 3] < 1]
    points_opengl = position[..., :3].reshape(-1, 3)
    # points_color = rgba[position[..., 3] < 1][..., :3]
    # Model matrix is the transformation from OpenGL camera space to SAPIEN world space
    # camera.get_model_matrix() must be called after scene.update_render()!
    model_matrix = camera.get_model_matrix()
    points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
    
    if coords == "world":
        return points_world
    elif coords == "camera":
        return points_opengl
    else:
        raise NotImplementedError("unknown coordinate!")
    

def get_depth_img(camera: sapien.CameraEntity, near: float = 0.1, far: float = 100) -> np.ndarray:
    position = camera.get_float_texture('Position')  # [H, W, 4]
    depth = -position[..., 2]
    # depth_image = (depth * (far / near)).astype(np.uint16)
    # print(depth_image.shape)
    return depth


def get_segment_img(camera: sapien.CameraEntity, level: str = "actor") -> np.ndarray:
    # ---------------------------------------------------------------------------- #
    # Segmentation labels
    # ---------------------------------------------------------------------------- #
    # Each pixel is (visual_id, actor_id/link_id, 0, 0)
    # visual_id is the unique id of each visual shape
    seg_labels = camera.get_uint32_texture('Segmentation')  # [H, W, 4]
    label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
    label1_image = seg_labels[..., 1].astype(np.uint8)  # actor-level

    if level == "actor":
        return label1_image
    elif level == "mesh":
        return label0_image
    else:
        raise NotImplementedError("unkown segment level!")
    

def spherical2cartesian(radius: float, theta: float, phi: float) -> np.ndarray:
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.array([x, y, z])


def cartesian2spherical(pos: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(pos)
    phi = np.arccos(pos[2] / r)
    theta = np.arctan(pos[1] / pos[0])
    return np.array([r, phi, theta])


def create_camera(scene: sapien.Scene, width: int, height: int, fovy: float, name: str) -> sapien.CameraEntity:
    # ---------------------------------------------------------------------------- #
    # Camera
    # ---------------------------------------------------------------------------- #
    near, far = 0.1, 100
    width, height = 640, 480
    camera = scene.add_camera(
        name=name,
        width=width,
        height=height,
        fovy=fovy,
        near=near,
        far=far,
    )
    camera.set_pose(sapien.Pose(p=[1, 0, 0]))

    return camera


def set_camera_local_pose(camera: sapien.CameraEntity, cam_vec: np.ndarray):
    # print(cam_vec)
    forward = -cam_vec / np.linalg.norm(cam_vec)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_vec
    # camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))
    camera.set_local_pose(sapien.Pose.from_transformation_matrix(mat44))


def camvec2pose(cam_vec: np.ndarray) -> sapien.Pose:
    forward = -cam_vec / np.linalg.norm(cam_vec)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_vec
    return sapien.Pose.from_transformation_matrix(mat44)


def generate_camera_pos_list(current_camera_pos: sapien.Pose, interpolation_sample_num: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    xyz_next = np.random.normal(loc=current_camera_pos.p, scale=0.03)
    quaternion_next = np.random.normal(loc=current_camera_pos.q, scale=0.01)
    quaternion_next = quaternion_next / np.linalg.norm(quaternion_next)
    xyz_inter = np.linspace(current_camera_pos.p, xyz_next, num=interpolation_sample_num)
    quaternion_inter = np.linspace(current_camera_pos.q, quaternion_next, num=interpolation_sample_num)
    return xyz_inter, quaternion_inter


def create_box(
        scene: sapien.Scene,
        pose: sapien.Pose,
        half_size: np.ndarray,
        render_material: sapien.RenderMaterial,
        name: str = '',
) -> sapien.Actor:
    """Create a box.

    Args:
        scene: sapien.Scene to create a box.
        pose: 6D pose of the box.
        half_size: [3], half size along x, y, z axes.
        color: [3] or [4], rgb or rgba
        name: name of the actor.

    Returns:
        sapien.Actor
    """
    half_size = np.array(half_size)
    builder: sapien.ActorBuilder = scene.create_actor_builder()
    builder.add_box_collision(half_size=half_size)  # Add collision shape
    builder.add_box_visual(half_size=half_size, material=render_material)  # Add visual shape
    
    box: sapien.Actor = builder.build_static(name=name)
    # Or you can set_name after building the actor
    # box.set_name(name)
    box.set_pose(pose)
    return box


def init_scene(ray_tracing: bool) -> Tuple[sapien.Scene, List[sapien.CameraEntity], List[np.ndarray]]:
    engine = sapien.Engine()
    if ray_tracing:
        sapien.render_config.camera_shader_dir = "rt"
        sapien.render_config.viewer_shader_dir = "rt"
        sapien.render_config.rt_samples_per_pixel = 256  # change to 256 for less noise
        # sapien.render_config.rt_max_path_depth = 16
        sapien.render_config.rt_use_denoiser = True  # change to True for OptiX denoiser
    renderer = sapien.SapienRenderer(offscreen_only=True)
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene_config.gravity = np.zeros(3)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

    ground_material = renderer.create_material()
    ground_material.base_color = np.array([202, 164, 114, 256]) / 256
    ground_material.specular = 0.5
    ground_material.set_diffuse_texture_from_file("ground.png")
    scene.add_ground(altitude=0, render_material=ground_material, render_half_size=np.array([10, 10]))

    wall_material = renderer.create_material()
    wall_material.set_diffuse_texture_from_file("wall.jpg")
    wall_front = create_box(scene, sapien.Pose(p=[10, 0, 0]), np.array([0.5, 10, 10]), wall_material, name='wall_front')
    wall_back = create_box(scene, sapien.Pose(p=[-10, 0, 0]), np.array([0.5, 10, 10]), wall_material, name='wall_back')
    wall_left = create_box(scene, sapien.Pose(p=[0, 10, 0]), np.array([10, 0.5, 10]), wall_material, name='wall_left')
    wall_right = create_box(scene, sapien.Pose(p=[0, -10, 0]), np.array([10, -0.5, 10]), wall_material, name='wall_right')

    radius1 = 2.5
    theta1 = np.deg2rad(195)
    phi1 = np.deg2rad(35)
    cam_vec1 = spherical2cartesian(radius1, theta1, phi1)
    # print(cam_pos1)

    camera1 = create_camera(scene, 640, 480, np.deg2rad(35), "camera1")

    radius2 = 2.5
    theta2 = np.deg2rad(165)
    phi2 = np.deg2rad(40)
    cam_vec2 = spherical2cartesian(radius2, theta2, phi2)
    # print(cam_pos2)

    camera2 = create_camera(scene, 640, 480, np.deg2rad(35), "camera2")

    return scene, [camera1, camera2], [cam_vec1, cam_vec2]


def render_surface(scene: sapien.Scene, obj_urdf_path: str, joint_data_root_dir: str, render_choice: str):
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    # load as a kinematic articulation
    if not os.path.exists(joint_data_root_dir):
        return
    with open(f"{joint_data_root_dir}/actor_pose.pkl", 'rb') as f_actor:
        actor_pose_dict = pickle.load(f_actor)
    init_base_pose = actor_pose_dict["actor_6"][0]
    actor_translation = init_base_pose[:3]
    # print("actor translation:", actor_translation)
    # actor_rotation = R.from_quat(init_base_pose[3:], scalar_first=True)
    articulate_obj = loader.load(obj_urdf_path)
    # q_90 = R.from_euler('z', -90, degrees=True).as_quat(scalar_first=True)
    articulate_obj.set_pose(sapien.Pose(init_base_pose[:3], q=init_base_pose[3:]))
    print("articulate obj pose:", articulate_obj.get_pose())
    # transform_pose = sapien.Pose(p=np.array([2, -3, boundingbox_min[2] + 0.5]), q=q_90)
    assert articulate_obj, 'URDF not loaded.'

    radius = 2.5
    theta = np.deg2rad(0)
    phi_angle_list = [np.deg2rad(30), np.deg2rad(60), np.deg2rad(90)]
    # phi_angle_list = [np.deg2rad(phi) for phi in range(20, 91, 10)]
    view_num_list = [8] * len(phi_angle_list)  # number of views for each phi angle
    camera = create_camera(scene, 640, 480, np.deg2rad(35), "camera")
    q_pos = np.load(f"{joint_data_root_dir}/qpos.npy")
    articulate_obj.set_qpos(q_pos[0])
    scene.step()
    
    img_list = []
    segment_list = []
    camera_list = []
    for round, phi in enumerate(phi_angle_list):
        for view_num in range(view_num_list[round]):
            new_theta = theta + view_num * np.deg2rad(360 / view_num_list[round])
            new_cam_vec = spherical2cartesian(radius, new_theta, phi)
            set_camera_local_pose(camera, new_cam_vec)
            new_pose = sapien.Pose(p=new_cam_vec + actor_translation, q=camera.pose.q)
            camera.set_local_pose(new_pose)

            # scene.step()  # make everything set
            scene.update_render()

            # save camera observation
            camera.take_picture()
            if render_choice == "xyz":
                img = get_xyz_img(camera)
                segment = get_segment_img(camera)
                segment_list.append(segment)
            elif render_choice == "rgb":
                img = get_rgba_img(camera)
                current_camera_pos = camera.get_pose().to_transformation_matrix()
                current_camera_pos = current_camera_pos[:, [1, 2, 0, 3]] * np.array([-1, 1, -1, 1])
                camera_list.append(current_camera_pos)
            img_list.append(img)
    # save view data
    view_data_dir = f"{joint_data_root_dir}/view_init/"
    os.makedirs(view_data_dir, exist_ok=True)

    for id in range(len(img_list)):
        if render_choice == "xyz":
            xyz_dir = f"{view_data_dir}/xyz"
            segment_dir = f"{view_data_dir}/segment"
            os.makedirs(xyz_dir, exist_ok=True)
            os.makedirs(segment_dir, exist_ok=True)
            np.savez_compressed(os.path.join(xyz_dir, "%06d.npz" % id), a=img_list[id])
            np.savez_compressed(os.path.join(segment_dir, "%06d.npz" % id), a=segment_list[id])
        elif render_choice == "rgb":
            rgb_dir = f"{view_data_dir}/rgb"
            os.makedirs(rgb_dir, exist_ok=True)
            sample_img_pil = Image.fromarray(img_list[id])
            sample_img_rgb = sample_img_pil.convert('RGBA')
            sample_img_rgb.save(os.path.join(rgb_dir, "%06d.png" % id))
            camera_dir = f"{view_data_dir}/camera/"
            os.makedirs(camera_dir, exist_ok=True)
            np.savez_compressed(os.path.join(camera_dir, f"{id:06d}.npz"), a=camera_list[id])

    scene.remove_articulation(articulate_obj)


def render_rgb(scene: sapien.Scene, camera_list: List[sapien.CameraEntity], obj_urdf_path: str,  obj_bounding_box: Dict[str, List[float]], 
               joint_id_list: List[int], joint_type_list: List[str], obj_data_root_dir: str, skip_exist: bool):
    boundingbox_min = np.array(obj_bounding_box["min"])
    boundingbox_max = np.array(obj_bounding_box["max"])

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    # load as a kinematic articulation
    articulate_obj = loader.load(obj_urdf_path)
    q_90 = R.from_euler('z', -90, degrees=True).as_quat(scalar_first=True)
    obj_pose = sapien.Pose(p=np.array([2, -3, -boundingbox_min[2] + 0.5]), q=q_90)
    articulate_obj.set_pose(obj_pose)
    assert articulate_obj, 'URDF not loaded.'

    joints, joint_names = get_joints_dict(articulate_obj)
    joint_limits = articulate_obj.get_qlimits()
    link_list = articulate_obj.get_links()
    for joint_id, joint_type in zip(joint_id_list, joint_type_list):
        if not os.path.exists(f"{obj_data_root_dir}/joint_{joint_id}"):
            break
        # print("processing joint", joint_id)
        joint_data_dir = f"{obj_data_root_dir}/joint_{joint_id}_bg/"
        if os.path.exists(joint_data_dir):
            if skip_exist:
                continue
            else:
                shutil.rmtree(joint_data_dir)

        # prepare joints
        control_joint_name = f"joint_{joint_id}"
        control_joint_id = joint_names.index(control_joint_name)
        init_qpos = np.zeros((joint_limits.shape[0],)) - 10
        init_qpos[control_joint_id] = joint_limits[control_joint_id, 0]
        init_qpos = np.clip(init_qpos, joint_limits[:, 0], joint_limits[:, 1])
        articulate_obj.set_qpos(init_qpos)
        print("joint limits:", joint_limits)
        control_joint = joints[control_joint_name]
        child_control_link = control_joint.get_child_link()
        link_base_pose = child_control_link.get_pose()
        cmass_global_pose = link_base_pose.transform(child_control_link.get_cmass_local_pose())
        transform_pose = sapien.Pose(p=cmass_global_pose.p, q=q_90)

        target_jpos = joint_limits[control_joint_id, 1]
        joints[control_joint_name].set_drive_property(stiffness=20, damping=5)
        joints[control_joint_name].set_drive_velocity_target(0.05)
        joints[control_joint_name].set_drive_target(target_jpos)

        plan_camera_pose_list = []
        for cam_id, camera in enumerate(camera_list):
            camera_pose = np.load(os.path.join(obj_data_root_dir, f"joint_{joint_id}/view_{cam_id}/camera_pose.npy"))
            plan_camera_pose_list.append(camera_pose)
            
            # print(plan_camera_pose_list[cam_id][0][:3])
            base_cam_spherical = cartesian2spherical(plan_camera_pose_list[cam_id][0][:3])

            r_xy = np.linalg.norm(plan_camera_pose_list[cam_id][0][:2])
            r_near = base_cam_spherical[0] / 2
            phi_high = base_cam_spherical[1] - 10 / 180 * np.pi
            new_r_xy = r_near * np.sin(phi_high)
            ratio = new_r_xy / r_xy
            base_cam_vec = np.array([ratio * plan_camera_pose_list[cam_id][0][0], ratio * plan_camera_pose_list[cam_id][0][1], r_near * np.cos(phi_high)])

            base_cam_pose = camvec2pose(base_cam_vec)
            
            transform_new_pose = transform_pose.transform(base_cam_pose)
            camera.set_local_pose(transform_new_pose)
            scene.update_render()

        # simulate and record video
        target_qpos = init_qpos.copy()
        target_qpos[control_joint_id] = target_jpos
        img_list = [[] for _ in range(len(camera_list))]
        camera_pose_list = [[] for _ in range(len(camera_list))]
        qpos_list = []
        actor_pose_dict = {}
        for actor in link_list:
            actor_id = actor.get_id()
            actor_pose_dict[f"actor_{actor_id}"] = []
        
        interpolation_sample_num = 10
        tmp_camera_pose_q = [Queue(maxsize=interpolation_sample_num) for _ in range(len(camera_list))]
        current_qpos = articulate_obj.get_qpos()
        previous_qpos = np.ones_like(current_qpos)
        step = 0
        max_step = 1000
        # max_step = np.random.randint(60, 90)
        plan_qpos_list = np.linspace(current_qpos, target_qpos, max_step)
        start_time = time.time()
        while (step < max_step) and \
        ((not np.allclose(current_qpos[control_joint_id], target_qpos[control_joint_id], atol=1e-5)) or \
         (not np.allclose(current_qpos[control_joint_id], previous_qpos[control_joint_id], atol=1e-4))):
        # for step in range(plan_qpos_list.shape[0]):
            articulate_obj.set_qpos(plan_qpos_list[step])
            previous_qpos = current_qpos.copy()

            # save camera observation
            obs_start = time.time()
            for cam_id, camera in enumerate(camera_list):
                if step % interpolation_sample_num == 0: # time to set new camera pose target
                    assert tmp_camera_pose_q[cam_id].empty(), f"camera{cam_id}'s pose queue not empty"
                    current_camera_pos = camera.get_pose()
                    xyz_inter, quaternion_inter = generate_camera_pos_list(current_camera_pos, interpolation_sample_num)
                    for sample_id in range(interpolation_sample_num):
                        tmp_camera_pose_q[cam_id].put(sapien.Pose(p=xyz_inter[sample_id], q=quaternion_inter[sample_id]))

                next_camera_pose = tmp_camera_pose_q[cam_id].get()
                camera.set_local_pose(next_camera_pose)

            scene.step()  # make everything set
            scene.update_render()

            for cam_id, camera in enumerate(camera_list):
                camera.take_picture()

                rgba_img = get_rgba_img(camera)
                img_list[cam_id].append(rgba_img)
                current_camera_pos = camera.get_pose()
                camera_pose_list[cam_id].append(np.concatenate([current_camera_pos.p, current_camera_pos.q]))
            
            obs_end = time.time()
            print("step:", step, "time for obs:", obs_end - obs_start)

            # prepare for next step
            current_qpos = articulate_obj.get_qpos()
            step += 1
            # if step == plan_qpos_list.shape[0]:
            #     break
            # save object state
            qpos_list.append(articulate_obj.get_qpos())
            for actor in link_list:
                actor_id = actor.get_id()
                actor_pose_dict[f"actor_{actor_id}"].append(np.concatenate([actor.get_pose().p, actor.get_pose().q]))

        end_time = time.time()
        print("time usage:", end_time - start_time)
        print("use steps:", step)

        # data validity check
        data_frame_len = len(qpos_list)
        for actor in link_list:
            actor_id = actor.get_id()
            assert len(actor_pose_dict[f"actor_{actor_id}"]) == data_frame_len, f"actor {actor_id} pose list length wrong"
        for cam_id in range(len(camera_list)):
            assert len(img_list[cam_id]) == data_frame_len, f"camera {cam_id} rgb list length wrong"
            assert len(camera_pose_list[cam_id]) == data_frame_len, f"camera {cam_id} camera pose list length wrong"

        # save data
        image_limits = 200
        # save joint data
        os.makedirs(joint_data_dir)
        meta_data = {"joint_id": int(joint_id), 
                     "init": float(joint_limits[control_joint_id, 0]), 
                     "target": float(joint_limits[control_joint_id, 1]), 
                     "time_usage": float(end_time - start_time), 
                     "simulation_steps": int(step)}
        with open(os.path.join(joint_data_dir, "meta.json"), 'w') as f_meta:
            json.dump(meta_data, f_meta)
        sample_qpos_list = qpos_list[::data_frame_len // image_limits + 1]
        sample_qpos_list = np.stack(sample_qpos_list, axis=0)
        np.save(os.path.join(joint_data_dir, "qpos.npy"), sample_qpos_list)
        for actor in link_list:
            actor_id = actor.get_id()
            actor_pose_dict[f"actor_{actor_id}"] = actor_pose_dict[f"actor_{actor_id}"][::data_frame_len // image_limits + 1]
        with open(os.path.join(joint_data_dir, "actor_pose.pkl"), "wb") as f_actor:
            pickle.dump(actor_pose_dict, f_actor)

        # save view data
        for cam_id in range(len(camera_list)):
            view_data_dir = f"{joint_data_dir}/view_{cam_id}/"
            os.mkdir(view_data_dir)
            img_dir = f"{view_data_dir}/rgb/"
            os.mkdir(img_dir)

            sample_img_list = img_list[cam_id][::data_frame_len // image_limits + 1]
            sample_camera_pose_list = camera_pose_list[cam_id][::data_frame_len // image_limits + 1]
            for id in range(len(sample_img_list)):
                sample_img_pil = Image.fromarray(sample_img_list[id])
                sample_img_rgb = sample_img_pil.convert('RGB')
                sample_img_rgb.save(os.path.join(img_dir, "%06d.jpg" % id))
            sample_camera_pose_list = np.stack(sample_camera_pose_list, axis=0)
            np.save(os.path.join(view_data_dir, "camera_pose.npy"), sample_camera_pose_list)
            pics2video(img_dir, f"{view_data_dir}/video.mp4")

        # reset joint
        # joints[control_joint_name].set_drive_target(plan_qpos_list[0])
    scene.remove_articulation(articulate_obj)


def render_depth_segment(scene: sapien.Scene, camera_list: List[sapien.CameraEntity], obj_urdf_path: str, joint_id_list: List[int], obj_data_root_dir: str):
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    # load as a kinematic articulation
    articulate_obj = loader.load(obj_urdf_path)
    # q_90 = R.from_euler('z', -90, degrees=True).as_quat(scalar_first=True)
    with open(f"{obj_data_root_dir}/joint_{joint_id_list[0]}_bg/actor_pose.pkl", 'rb') as f_actor:
        actor_pose_dict = pickle.load(f_actor)
    init_base_pose = actor_pose_dict["actor_6"][0]
    articulate_obj.set_pose(sapien.Pose(init_base_pose[:3], q=init_base_pose[3:]))
    print("articulate obj pose:", articulate_obj.get_pose())
    # articulate_obj.set_pose(obj_pose)
    # transform_pose = sapien.Pose(p=np.array([2, -3, boundingbox_min[2] + 0.5]), q=q_90)
    assert articulate_obj, 'URDF not loaded.'

    link_list = articulate_obj.get_links()
    for link in link_list:
        print("link actor id", link.get_id())

    for joint_id in joint_id_list:
        joint_data_dir = f"{obj_data_root_dir}/joint_{joint_id}_bg/"
        if not os.path.exists(joint_data_dir):
            continue

        qpos_list = np.load(os.path.join(joint_data_dir, "qpos.npy"))
        plan_camera_pose_list = []
        for cam_id, camera in enumerate(camera_list):
            camera_pose = np.load(os.path.join(joint_data_dir, f"view_{cam_id}/camera_pose.npy"))
            plan_camera_pose_list.append(camera_pose)
        
        segment_list = [[] for _ in range(len(camera_list))]
        xyz_list = [[] for _ in range(len(camera_list))]
        depth_list = [[] for _ in range(len(camera_list))]
        for step in range(qpos_list.shape[0]):
            articulate_obj.set_qpos(qpos_list[step])
            for cam_id, camera in enumerate(camera_list):
                cam_pose = sapien.Pose(plan_camera_pose_list[cam_id][step][:3], plan_camera_pose_list[cam_id][step][3:])
                camera.set_local_pose(cam_pose)
            scene.step()  # make everything set
            scene.update_render()

            for cam_id, camera in enumerate(camera_list):
                camera.take_picture()
                segment_img = get_segment_img(camera)
                segment_list[cam_id].append(segment_img)
                xyz_img = get_xyz_img(camera)
                xyz_list[cam_id].append(xyz_img)
                depth_img = get_depth_img(camera)
                depth_list[cam_id].append(depth_img)

        # data validity check
        data_frame_len = len(qpos_list)
        for cam_id in range(len(camera_list)):
            assert len(segment_list[cam_id]) == data_frame_len, f"camera {cam_id} segment list length wrong"
            assert len(xyz_list[cam_id]) == data_frame_len, f"camera {cam_id} xyz list length wrong"
            assert len(depth_list[cam_id]) == data_frame_len, f"camera {cam_id} depth list length wrong"

        # save segment data
        for cam_id in range(len(camera_list)):
            view_data_dir = f"{joint_data_dir}/view_{cam_id}/"
            np.save(f"{view_data_dir}/intrinsics.npy", camera_list[cam_id].get_intrinsic_matrix())
            xyz_dir = f"{view_data_dir}/xyz/"
            if os.path.exists(xyz_dir):
                shutil.rmtree(xyz_dir)
            os.mkdir(xyz_dir)
            segment_dir = f"{view_data_dir}/segment/"
            if os.path.exists(segment_dir):
                shutil.rmtree(segment_dir)
            os.mkdir(segment_dir)
            depth_dir = f"{view_data_dir}/depth/"
            if os.path.exists(depth_dir):
                shutil.rmtree(depth_dir)
            os.mkdir(depth_dir)
            for frame_id in range(data_frame_len):
                if frame_id == 0:
                    np.savez_compressed(os.path.join(xyz_dir, "%06d.npz" % frame_id), a=xyz_list[cam_id][frame_id])
                np.savez_compressed(os.path.join(segment_dir, "%06d.npz" % frame_id), a=segment_list[cam_id][frame_id])
                np.savez_compressed(os.path.join(depth_dir, "%06d.npz" % frame_id), a=depth_list[cam_id][frame_id])
            
    scene.remove_articulation(articulate_obj)


def main(dataset_config_path: str, partnet_mobility_dir: str, data_root_path: str, ray_tracing: bool, render_choice: str):
    os.makedirs(data_root_path, exist_ok=True)
    with open(dataset_config_path, 'r') as f_dataset:
        dataset_config_dict = json.load(f_dataset)
    for model_cat in dataset_config_dict.keys():
        print("processing", model_cat)
        model_data_dir = f"{data_root_path}/{model_cat}"
        os.makedirs(model_data_dir, exist_ok=True)
        cat_dict = dataset_config_dict[model_cat]
        obj_list = list(cat_dict.keys())
        for obj_id in tqdm(obj_list):
            obj_urdf_file_path = f"{partnet_mobility_dir}/{obj_id}/mobility.urdf"
            boundingbox = cat_dict[obj_id]["boundingbox"]
            joint_list = []
            joint_type_list = []
            for interaction in cat_dict[obj_id]["interaction_list"]:
                joint_list.append(interaction["id"])
                joint_type_list.append(interaction["type"])
            obj_data_dir = f"{model_data_dir}/{obj_id}"
            if not os.path.exists(obj_data_dir):
                continue
            scene, camera_list, cam_vec_list = init_scene(ray_tracing=ray_tracing)

            if render_choice == "depth":
                render_depth_segment(scene, camera_list, obj_urdf_file_path, joint_list, obj_data_dir)
            elif render_choice == "rgb":
                render_rgb(scene, camera_list, obj_urdf_file_path, boundingbox, joint_list, joint_type_list, obj_data_dir, skip_exist=False)
            for joint_id in joint_list:
                render_surface(scene, obj_urdf_file_path, f"{obj_data_dir}/joint_{joint_id}_bg", render_choice)

            del scene
            del camera_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Render interaction simulation for PartNet Mobility dataset.")
    parser.add_argument("--meta_file_path", type=str, default="new_partnet_mobility_dataset_correct_intr_meta.json")
    parser.add_argument("--partnet_mobility_path", type=str, default="partnet-mobility-v0/")
    parser.add_argument("--data_root_path", type=str, default="sim_data/partnet_mobility/")
    args = parser.parse_args()
    main(args.meta_file_path, args.data_root_path, True, "rgb")
    main(args.meta_file_path, args.data_root_path, False, "xyz")
