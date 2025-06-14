import zarr
import os
import json
import glob
import numpy as np
from PIL import Image
import fpsample
import open3d as o3d
import torch
from tqdm import tqdm
import sys
import argparse
sys.path.append(os.getcwd()) 
from utils.forward_kinematics import forward_kinematics
from utils.vis import vis_point_cloud

def contains_zarr(s):
    return 'zarr' in s

def sample_point_cloud(point, mask, num_point):

    samples_idx = fpsample.bucket_fps_kdline_sampling(point[:,:3], num_point,h=7)
    point = point[samples_idx]
    mask = mask[samples_idx]

    return point, mask

def uniform_sampling(array, num):

    start = 0
    stop = array.shape[0]-1
    sample_id = np.int_(np.linspace(start, stop, num))
    

    sample_array = array[sample_id]

    return sample_array

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--save_name', type=str)
    parser.add_argument('--vis', action="store_false", help='whether or not visualize affordance')
    args = parser.parse_args()

    return args

def process_data(data_dir=None, save_dir=None, save_name=None, vis=False):
    root = data_dir

    state_arrays = []
    action_arrays = []
    point_arrays = []
    mask_arrays = []
    img_arrays = []
    afford_arrays = []
    episode_ends = []
    pose_arrays = []
    init_pos_arrays = []
    total = 0 

    for data_root in os.listdir(root):
        
        if contains_zarr(data_root):
            continue

        for data_dir in tqdm(os.listdir(f"{root}/{data_root}")):
            
            if data_dir == 'config.json':
                continue

            data_root_abs = f"{root}/{data_root}"
            
            state_array = []
            action_array = []
            point_array = []
            raw_point_array = []
            mask_array = []
            raw_mask_array = []
            afford_ = []
            afford_array = []
            img_array = []
            pose_array = []
            init_pos_array = []

            data_files = glob.glob(os.path.join(data_root_abs, data_dir, "*.npz"))
            data_files.sort(key=lambda x:int(x.split('/')[-1].split('.')[0]))

            json_file = os.path.join(data_root_abs, data_dir, "closed_gripper.json")
            with open(json_file, 'r', encoding='utf-8') as file:
                closed_gripper_step = json.load(file)['closed_gripper']


            for id, data_file in enumerate(data_files):
                
                if id%2==0:

                    data = np.load(data_file)
                    state = data['state']
                    action = data['action']
                    point = data['point']
                    mask = data['mask']
                    init_pos = data['init_position']
                    pose = data['franka_init_pose']
                    
                    if id == 0:
                        afford = init_pos
                        afford_.append(afford.reshape(1,-1))
                    if id > closed_gripper_step:
                        robot_qpos = state[0][-9:]
                        afford = forward_kinematics(robot_qpos, np.array(pose[:3]), dist=-0.02)
                        afford_.append(afford.reshape(1,-1))
                    downsample_point, dowmsample_mask = sample_point_cloud(point, mask, 4096)
                    raw_point, raw_mask = sample_point_cloud(point, mask, 80000)

                    img_file = f'step-{str(id).zfill(4)}.png'
                    original_image = Image.open(os.path.join(data_root_abs,data_dir,'video',img_file))
                    original_image = original_image.convert("RGB")
                    resized_img = original_image.resize((320,240))
                    # resized_img.save('output_img.png')
                    img = np.array(resized_img)

                    state_array.append(state)
                    init_pos_array.append(init_pos.reshape(1,-1))
                    action_array.append(action)
                    point_array.append(downsample_point[np.newaxis,:,:])
                    raw_point_array.append(raw_point[np.newaxis,:,:])
                    mask_array.append(dowmsample_mask[np.newaxis,:,:])
                    raw_mask_array.append(raw_mask[np.newaxis,:,:])
                    img_array.append(img[np.newaxis,:,:,:])
                    pose_array.append(pose.reshape(1,-1))


            state_array = np.concatenate(state_array,axis=0)
            action_array = np.concatenate(action_array,axis=0)
            point_array = np.concatenate(point_array,axis=0)
            raw_point_array = np.concatenate(raw_point_array, axis=0)
            mask_array = np.concatenate(mask_array,axis=0)
            raw_mask_array = np.concatenate(raw_mask_array,axis=0)
            img_array = np.concatenate(img_array,axis=0)
            init_pos_array = np.concatenate(init_pos_array, axis=0)
            pose_array = np.concatenate(pose_array,axis=0)

            afford_ = np.concatenate(afford_, axis=0)
            afford_ = uniform_sampling(afford_, num=8)[np.newaxis,:,:]

            if vis:
                afford_color = np.zeros_like(afford_[0])
                afford_color[:,0] = 255
                afford_pc = np.concatenate((afford_[0], afford_color), axis=1)
                pc = np.concatenate((point_array[0], afford_pc), axis=0)
                vis_point_cloud(pc, f"{data_root_abs}/{data_dir}/pc_afford")

            afford_array = np.repeat(afford_, len(state_array),axis=0)
            np.savez(f"{data_root_abs}/{data_dir}.npz", state = state_array,
                        action = action_array,
                        raw_point = raw_point_array,
                        raw_mask = raw_mask_array,
                        img = img_array,
                        afford = afford_array,
                        pose = pose_array)

            total += len(state_array)
            state_arrays.append(state_array)
            action_arrays.append(action_array)
            point_arrays.append(point_array)
            mask_arrays.append(mask_array)
            img_arrays.append(img_array)
            afford_arrays.append(afford_array)
            pose_arrays.append(pose_array)
            init_pos_arrays.append(init_pos_array)
            episode_ends.append(total)

    state_arrays = np.concatenate(state_arrays,axis=0)
    action_arrays = np.concatenate(action_arrays,axis=0)
    point_arrays = np.concatenate(point_arrays,axis=0)
    mask_arrays = np.concatenate(mask_arrays,axis=0)
    img_arrays = np.concatenate(img_arrays,axis=0)
    afford_arrays = np.concatenate(afford_arrays,axis=0)
    pose_arrays = np.concatenate(pose_arrays,axis=0)
    init_pos_arrays = np.concatenate(init_pos_arrays, axis=0)

    print("state:",state_arrays.shape)
    print("action:",action_arrays.shape)
    print("point",point_arrays.shape)
    print("mask:",mask_arrays.shape)
    print("img:",img_arrays.shape)
    print("afford:",afford_arrays.shape)
    print("pose:",pose_arrays.shape)
    print("init_pos:", init_pos_arrays.shape)
    print("total_frame:",episode_ends[-1])

    print("=============================")

    print("begining to create datasets!")
    group_file = f"{save_dir}/{save_name}.zarr"
    zarr_root = zarr.group(group_file)
    try:
        zarr_data = zarr_root['data']
        print("Group 'data' exists")
    except KeyError:
        zarr_data = zarr_root.create_group('data')
    try:
        zarr_meta = zarr_root['meta']
        print("Group 'meta' exists")
    except KeyError:
        zarr_meta = zarr_root.create_group('meta')

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    action_chunk_size = (250,action_arrays.shape[1])
    state_chunk_size = (250,state_arrays.shape[1])
    afford_chunk_size = (250,afford_arrays.shape[1],afford_arrays.shape[2])
    pose_chunk_size = (250,pose_arrays.shape[1])
    point_chunk_size = (250,point_arrays.shape[1],point_arrays.shape[2])
    mask_chunk_size = (250,mask_arrays.shape[1],mask_arrays.shape[2])
    img_chunk_size = (250,img_arrays.shape[1],img_arrays.shape[2],img_arrays.shape[3]) 
    init_pos_chunk_size = (250, init_pos_arrays.shape[1])

    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('mask', data=mask_arrays, chunks=mask_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point', data=point_arrays, chunks=point_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('afford', data=afford_arrays, chunks=afford_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('pose', data=pose_arrays, chunks=pose_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('init_pos', data=init_pos_arrays, chunks=init_pos_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends, dtype='int64', overwrite=True, compressor=compressor)
    
    
if "__main__" ==__name__:
    
    args = parse_args() 
    process_data(data_dir=args.data_dir, save_dir=args.save_dir, save_name=args.save_name, vis=args.vis)