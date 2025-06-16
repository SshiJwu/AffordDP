from isaacgym import gymapi
import tqdm
import numpy as np
import fpsample
import torch
import os
import random
import open3d as o3d
from afforddp.common.pytorch_util import dict_apply
from afforddp.env_runner.base_pointcloud_runner import BasePointcloudRunner
from afforddp.gym_util.multistep_wrapper import MultiStepWrapper
from afforddp.env.vec_task import VecTaskPython
from afforddp.env.CabinetManip import CabinetManipEnv
from afforddp.gym_util.utils import read_yaml_config
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy.ndimage import label
from afforddp.featurizer.utils.visualization import IMG_SIZE
from afforddp.retrieval.retrieval_buf import RetrievalBuf
from afforddp.retrieval.affordance_transfer import affordance_transfer
from afforddp.utils.vision_model import run_pointsam, run_sam, scale_img_pixel, transfer_pixel
from afforddp.utils.transform import get_image_pixel_from_3d, sample_point_cloud, find_nearest_object_pixel_in_box, get_3d_from_image_pixel, ICP_register,  update_afford
from afforddp.utils.vis import show_results, vis_point_cloud 
from third_party.Point_SAM.demo.utils import draw
import shutil
from datetime import datetime

class CabinetManipRunner(BasePointcloudRunner):

    def __init__(self,
                n_eval=10,
                max_steps=250,
                n_obs_steps=8,
                n_action_steps=8,
                task_name=None,
                data_dir=None,
                memory_dir=None,
                output_dir=None,
                rl_device=None,
                clip_actions=None,
                clip_observations=None,
                config_path=None,
                object_id=None):
        
        cfgs = read_yaml_config(config_path)
        cfgs['asset']['arti']['arti_gapartnet_ids'] = [object_id]

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps

        self.task_name = task_name
        self.data_dir = data_dir
        self.memory_dir = memory_dir
        self.episode_eval = n_eval
        self.object_id = object_id

        self.env = MultiStepWrapper(VecTaskPython(CabinetManipEnv(cfgs),rl_device, clip_observations, clip_actions),
                                n_obs_steps=n_obs_steps,
                                n_action_steps=n_action_steps,
                                max_episode_steps=max_steps,
                                reward_agg_method='sum')
        
        self.memory_buffer = RetrievalBuf(data_dir=self.data_dir, save_dir=self.memory_dir, task_name=task_name)

        self.output_dir =  f"{output_dir}/{self.task_name}/{self.object_id}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        else:
            shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir)

    def run(self, policy):
        
        device = policy.device
        env = self.env
        all_success_return = []
        success_pos = []
        
        self.results_file = os.path.join(self.output_dir, "results.txt")
        with open(self.results_file, 'w') as f:
            f.write(f"Evaluation Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Task: {self.task_name}, Object ID: {self.object_id}, Num Eval: {self.episode_eval}\n")
            f.write("-"*50 + "\n")
            f.write("Episode\tSuccess\trun_step\n")
            
        for episode_id in tqdm.tqdm(range(self.episode_eval), desc=f"{self.task_name} Eval Env",leave=False):
            
            save_dir = os.path.join(self.output_dir, str(episode_id))
            obs = env.reset()
            env.task.gym.clear_lines(env.task.viewer)
            env.task.get_gapartnet_anno()
            env.task.cal_handle(bbox_id=-1)
            policy.reset()
            franka_base_pos = env.task.franka_reset_pos_list[0]
            
            transfer_afford = affordance_transfer(prompt='cabinet', 
                                                    gym=env.task, 
                                                    save_dir=f"{self.output_dir}/vis/{episode_id}", 
                                                    memory_buffer=self.memory_buffer, 
                                                    task_name=self.task_name)

            for step_id in range(self.max_steps):

                # create obs dict
                np_obs_dict = {
                    'point_cloud': np.array(obs[:,0])[:,:4096,:3],
                    'state': np.array(obs[:,0])[:,4096:,0]
                }

                obs_dict = dict_apply(np_obs_dict,
                                lambda x: torch.from_numpy(x).to(
                                    device=device))
                
                obs_dict_input = {}  
                obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                obs_dict_input['state'] = obs_dict['state'].unsqueeze(0)
                afford = transfer_afford
                afford = torch.from_numpy(afford.reshape(1,*afford.shape)).to(device=device)
                afford = afford.repeat(self.n_obs_steps,1,1)
                action_dict = policy.predict_action(obs_dict_input, afford, base_pose=franka_base_pos)
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # path = os.path.join(save_dir,str(step_id)+".png")
                obs, success = env.step(action, save_dir, step_id)
                if success:
                    break

            all_success_return.append(success.item())
            result_line = f"[{episode_id}/{self.episode_eval}]:\t{success.item()}\t{step_id}\n"
            
            with open(self.results_file, 'a') as f:
                f.write(result_line)

        success_rate = all_success_return.count(1)/self.episode_eval
        with open(self.results_file, 'a') as f:
                f.write("---------------- Eval Results --------------\n")
                f.write(f"success rate:{success_rate}")




