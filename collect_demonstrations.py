from afforddp.env.CabinetManip import CabinetManipEnv
import numpy as np
from afforddp.gym_util.utils import read_yaml_config
import torch
import glob
from isaacgym import gymapi
import json
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import os
import sys
import tqdm
import os
import shutil
from isaacgym import gymutil
import argparse

torch.set_printoptions(precision=4, sci_mode=False)

def save_config(data, save_config_path):

    with open(save_config_path, 'w') as file:
        json.dump(data, file, indent=4)
        
def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default='PullDrawer.yaml', help='Environment Cobfig Name')
    parser.add_argument('--save_dir', type=str, default='record')
    parser.add_argument('--obj_id', type=int, help='Gapartnet asset id')
    parser.add_argument('--part_id', type=int, default=-1, help='select part to manipulation')

    args = parser.parse_args()

    return args

def collect_demo(args):
    
    task_name = args.config_name.split(".")[0]
    if not task_name in ['PullDrawer', 'OpenDoor']:
        raise ValueError(f'Invalid task_type: {task_name}')
        
    config_path = os.path.join(os.getcwd(),"afforddp/config/env",args.config_name)
    cfgs = read_yaml_config(config_path)
    obj_id = args.obj_id
    cfgs['asset']['arti']['arti_gapartnet_ids'] = [obj_id]
    num_demos = cfgs['num_demos']

    gym = CabinetManipEnv(cfgs)
    
    save_data_dir = f"{args.save_dir}/{task_name}/{obj_id}"
    
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)
    
    config_path = f'{save_data_dir}/config.json'
    if not os.path.exists(config_path):
        save_config(cfgs, config_path)

    with tqdm.tqdm(total=num_demos) as pbar:
        
        pbar.set_description('Collecting demonstrations:')
        count = 0

        while(count!=num_demos):

            gym.reset(to_reset="all")
            # get the gapartnet annotation
            gym.get_gapartnet_anno()
            
            # render bbox for visualization and debug
            if not cfgs["HEADLESS"] and True:
                gym.gym.clear_lines(gym.viewer)

            gym.cal_handle(bbox_id=args.part_id)
            save_root=f'{save_data_dir}/traj_{count}'


            if not os.path.exists(save_root):
                os.makedirs(save_root)
                success = gym.motion_planning(save_video=True, save_root=save_root, task_type=task_name)
                if not success:
                    shutil.rmtree(save_root)
                else:
                    count += success
                    pbar.update(1)
                    
            else:
                save_path = save_root.split("/")[:-1]
                path = os.listdir('/'.join(save_path))
                path = [item for item in path if item != "config.json"]
                path.sort(key=lambda x:int(x.split('_')[-1]))
                traj_id = int(path[-1].split('_')[-1]) + 1
                save_root = f'{save_data_dir}/traj_{traj_id}'
                os.makedirs(save_root)
                success = gym.motion_planning(save_video=True, save_root=save_root, task_type=task_name)
                if not success:
                    shutil.rmtree(save_root)
                else:
                    count += success
                    pbar.update(1)

    del gym

if __name__ =='__main__':
    args = parse_args()
    collect_demo(args)
