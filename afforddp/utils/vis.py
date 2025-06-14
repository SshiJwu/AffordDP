from PIL import ImageDraw, Image
from afforddp.featurizer.utils.visualization import IMG_SIZE, Demo
import torch
import open3d as o3d
import numpy as np
from afforddp.utils.vision_model import extract_ft

def show_results(PIL_img, points, save_name, color='red'):

    draw = ImageDraw.Draw(PIL_img)

    for point in points:
        draw.ellipse([point[0]-5, point[1]-5, point[0]+5, point[1]+5], fill=color)
    PIL_img.save(f'{save_name}.png')


def run_demo(src_path, tgt_path, prompt):
    file_list = [src_path, tgt_path]
    img_list = []
    ft_list = []
    for filename in file_list:
        img = Image.open(filename).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_list.append(img)
        ft = extract_ft(img, prompt)
        ft_list.append(ft)
    
    ft = torch.cat(ft_list, dim=0)
    demo = Demo(img_list, ft, IMG_SIZE)
    demo.plot_img_pairs(fig_size=5)

def vis_point_cloud(point_cloud_array, name="point_cloud_afford"):
    point_cloud_array = np.concatenate((point_cloud_array, np.array([0,0,0, 255, 0, 255]).reshape(1, -1)), axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_array[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud_array[:,3:6]/255)
    o3d.io.write_point_cloud(f"{name}.ply", pcd)
