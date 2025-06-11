from PIL import ImageDraw
import os
import cv2
import numpy as np
import open3d as o3d

def show_results(PIL_img, points, save_name, color='red'):

    draw = ImageDraw.Draw(PIL_img)

    for point in points:
        draw.ellipse([point[0]-5, point[1]-5, point[0]+5, point[1]+5], fill=color)
    PIL_img.save(f'{save_name}.png')

def save_img(images, demo_idx, root_dir=None):

    save_dir = f"{root_dir}/image/demo_{demo_idx}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    for i in range(images.shape[0]):
        image = images[i]
        cv2.imwrite(f'{save_dir}/{i}.jpg', image)

def save_video(img_array, demo_id, root_dir=None):

    fps = 30
    height, width, _ = img_array[0].shape
    save_dir = f'{root_dir}/video'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = f'{save_dir}/demo_{demo_id}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = cv2.VideoWriter(save_name, fourcc, fps, (width, height))

    for image in img_array:
        out.write(image)
    out.release()


def vis_point_cloud(point_cloud_array, name=None):
    point_cloud_array = np.concatenate((point_cloud_array, np.array([0,0,0, 255, 0, 255]).reshape(1, -1)), axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_array[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud_array[:,3:6]/255)
    o3d.io.write_point_cloud(f"{name}.ply", pcd)