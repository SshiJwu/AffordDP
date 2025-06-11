import cv2
import numpy as np
import os, glob
import argparse
import imageio
from PIL import Image
from isaacgym.torch_utils import *
import torch
import math
import yaml
import random
import sys
sys.path.append("../")
sys.path.append("../vision")
# from vision.grounded_sam_demo import prepare_GroundedSAM_for_inference

import open3d as o3d
import numpy as np


def generate_urdf(obj_name, obj_path, save_root):
    
    urdf_save_path = f"{save_root}/{obj_name}.urdf"

    if os.path.exists(f"{save_root}/{obj_path}"):
        # generate urdf content
        urdf_content = f"""<?xml version="1.0"?>
<robot name="{obj_name}">
<link name="{obj_name}">
    <visual>
    <origin xyz="0.0 0.0 0.0"/>
    <geometry>
        <mesh filename="{obj_path}" scale="1.0 1.0 1.0"/>
    </geometry>
    </visual>
    <collision>
    <origin xyz="0.0 0.0 0.0"/>
    <geometry>
        <mesh filename="{obj_path}" scale="1.0 1.0 1.0"/>
    </geometry>
    </collision>
    <inertial>
    <mass value="1.0"/>
    <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
</link>
</robot>
    """
    else:
        import pdb; pdb.set_trace()
    # Write the URDF file in the object's directory, not inside the google_16k directory
    with open(urdf_save_path, 'w') as file:
        file.write(urdf_content)

def get_rotation_from_obj_instruction(obj_path, obj_name, instruction, rotation_engine, save_root):
    # import pdb; pdb.set_trace()
    
    # obj_name = obj_path.split("/")[-1].split(".")[0]
    mtl_path = "/".join(obj_path.split("/")[:-1]) + "/material.mtl"
    png_path = "/".join(obj_path.split("/")[:-1]) + "/material_0.png"
    assert os.path.exists(mtl_path)
    assert os.path.exists(png_path)
    # import pdb; pdb.set_trace()
    os.system(f"cp {obj_path} {save_root}")
    os.system(f"cp {mtl_path} {save_root}")
    os.system(f"cp {png_path} {save_root}")
    generate_urdf(obj_name=obj_name, obj_path="material.obj", save_root = save_root)
    # import pdb; pdb.set_trace()
    final_rotations = rotation_engine.get_final_rotation(
        original_mesh_folder=save_root,
        asset_root=save_root,
        mesh_urdf_path=f"{obj_name}.urdf", #f"{obj_name}.urdf",
        instructions=[instruction], # specified label or arbitrary prompt
        #prompts = [], # would paralyze 'instructions' if 'prompts' is not empty
        output_folder=save_root,
        rendering_view=30,
        multi_image=True,
        experiment=False,
        ablate_engine=False
    )
    
    return final_rotations


def get_point_cloud_from_rgbd(depth, rgb, seg, vinv, proj, cam_w, cam_h):
    
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    # Ignore any points which originate from ground plane or empty space
    # depth_buffer[seg_buffer == 0] = -10001
    points = []
    colors = []

    centerU = cam_w/2
    centerV = cam_h/2
    for i in range(cam_w):
        for j in range(cam_h):
            if depth[j, i] < -10000:
                continue
            if seg == None or seg[j, i] > 0:
                u = -(i-centerU)/(cam_w)  # image-space coordinate
                v = (j-centerV)/(cam_h)  # image-space coordinate
                d = depth[j, i]  # depth buffer value
                X2 = [d*fu*u, d*fv*v, d, 1]  # deprojection vector
                p2 = X2*vinv  # Inverse camera view to get world coordinates
                points.append([p2[0, 0], p2[0, 1], p2[0, 2]])
                colors.append(rgb[j, i, :3])
    points, colors = np.array(points), np.array(colors)
    # import pdb; pdb.set_trace()
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])

    # point_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3]/255.0)

    # o3d.visualization.draw_geometries([point_cloud])
    return np.array(points), np.array(colors)

def rgbd_to_point_cloud(rgb_image_np, depth_image_np, width, height, fx, fy, cx, cy):
    # Load RGB and Depth images
    # For this example, let's assume you have RGB and Depth images as numpy arrays
    # rgb_image_np = <your RGB image as numpy array>
    # depth_image_np = <your Depth image as numpy array>

    # Convert numpy arrays to Open3D images
    rgb_image_o3d = o3d.geometry.Image(rgb_image_np)
    depth_image_o3d = o3d.geometry.Image(depth_image_np)

    # Create an RGBD image from the RGB and Depth images
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image_o3d, depth_image_o3d)

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)

    # Convert the RGBD image to a point cloud
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

    # Optionally visualize the point cloud
    # o3d.visualization.draw_geometries([point_cloud])

    return np.array(point_cloud.points), np.array(point_cloud.colors)

def get_point_cloud_from_rgbd_GPU(camera_depth_tensor, camera_rgb_tensor, camera_seg_tensor, camera_view_matrix_inv, camera_proj_matrix, width:float, height:float):
    # time1 = time.time()
    # print(u,v,width, height)
    # exit(123)
    device = camera_depth_tensor.device
    depth_buffer = camera_depth_tensor.to(device)
    rgb_buffer = camera_rgb_tensor.to(device)
    if camera_seg_tensor is not None:
        seg_buffer = camera_seg_tensor.to(device)

    # Get the camera view matrix and invert it to transform points from camera to world space
    vinv = torch.tensor(camera_view_matrix_inv).to(device)

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    
    proj = torch.tensor(camera_proj_matrix).to(device)
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]
    
    camera_u = torch.arange(0, width, device=device)
    camera_v = torch.arange(0, height, device=device)

    v, u = torch.meshgrid(
    camera_v, camera_u)

    centerU = width/2
    centerV = height/2

    Z = depth_buffer
    X = -(u-centerU)/width * Z * fu
    Y = (v-centerV)/height * Z * fv
    # print(rgb_buffer.shape)
    # print(seg_buffer.shape)
    R = rgb_buffer[...,0].view(-1)
    G = rgb_buffer[...,1].view(-1)
    B = rgb_buffer[...,2].view(-1)
    if camera_seg_tensor is not None:
        S = seg_buffer.view(-1)
        
    Z = Z.view(-1)
    X = X.view(-1)
    Y = Y.view(-1)

    if camera_seg_tensor is not None:
        position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device), R, G, B, S))
    else:
        position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device), R, G, B))
    position = position.permute(1, 0)
    position[:,0:4] = position[:,0:4]@vinv
    # print(position.shape)
    points = torch.cat((position[:, 0:3], position[:, 4:8]), dim = 1)

    return points

def images_to_video(image_folder, video_path, frame_size=(1920, 1080), fps=30):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg") or img.endswith(".jpeg")])
    if not images:
        print("No images found in the specified directory!")
        return
    
    writer = imageio.get_writer(video_path, fps=fps)
    
    for image in images:
        img_path = os.path.join(image_folder, image)
        img = imageio.imread(img_path)

        if img.shape[1] > frame_size[0] or img.shape[0] > frame_size[1]:
            # print("Warning: frame size is smaller than the one of the images.")
            # print("Images will be resized to match frame size.")
            img = np.array(Image.fromarray(img).resize(frame_size))
        
        writer.append_data(img)
    
    writer.close()
    print("Video created successfully!")
    
def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats

def read_yaml_config(file_path):
    with open(file_path, 'r') as file:
        # Load the YAML file into a Python dictionary
        config = yaml.safe_load(file)
    return config