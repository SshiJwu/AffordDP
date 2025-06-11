from isaacgym import gymapi
import tqdm
import numpy as np
import fpsample
import torch
import os
import random
import open3d as o3d
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_pointcloud_runner import BasePointcloudRunner
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.env.vec_task import VecTaskPython
from diffusion_policy_3d.env.DrawerManip import DrawerManipEnv
from diffusion_policy_3d.gym_util.utils import read_yaml_config
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy.ndimage import label
from diffusion_policy_3d.vision.featurizer.utils.visualization import IMG_SIZE
from diffusion_policy_3d.env_runner.retrieval import MemoryBank
from diffusion_policy_3d.utils.vision_model import run_pointsam, run_sam, scale_img_pixel, transfer_pixel
from diffusion_policy_3d.utils.transform import get_image_pixel_from_3d, sample_point_cloud, find_nearest_object_pixel_in_box, get_3d_from_image_pixel, ICP_register,  update_afford
from diffusion_policy_3d.utils.vis import show_results, vis_point_cloud 
from diffusion_policy_3d.vision.Point_SAM.demo.utils import draw

class DrawerManipRunner(BasePointcloudRunner):

    def __init__(self,
                n_eval=10,
                max_steps=250,
                n_obs_steps=8,
                n_action_steps=8,
                affordance=False,
                dynamic_affordance=False,
                task_name=None,
                memory_path=None,
                rl_device=None,
                clip_actions=None,
                clip_observations=None,
                config_path=None):
        

        cfgs = read_yaml_config(config_path)
        id = cfgs['asset']['arti']['arti_gapartnet_ids'][0]

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps

        self.task_name = task_name
        self.episode_eval = n_eval

        self.env = MultiStepWrapper(VecTaskPython(DrawerManipEnv(cfgs),rl_device, clip_observations, clip_actions),
                                n_obs_steps=n_obs_steps,
                                n_action_steps=n_action_steps,
                                max_episode_steps=max_steps,
                                reward_agg_method='sum')
        
        # self.memory_buffer = MemoryBank(memory_path)
                                
        self.affordance = affordance
        self.dynamic_affordance = dynamic_affordance


        output_dir =  f"vis/{id}/{self.task_name}_{self.affordance}_{self.dynamic_affordance}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        super().__init__(output_dir)

        if self.affordance:
            self.memory_buffer = MemoryBank(memory_path)
            self.retrieval_path = f"{memory_path}/{task_name}"

    def run(self, policy):
        
        device = policy.device
        env = self.env
        all_success_return = []
        retrieval_id_list = []
        success_pos = []
        
        
        for episode_id in tqdm.tqdm(range(self.episode_eval), desc=f"{self.task_name} Eval Env",leave=False):
            
            policy.DSG_threshold = 0.15
            save_dir = os.path.join(self.output_dir,str(episode_id))
            obs = env.reset()
            env.task.gym.clear_lines(env.task.viewer)
            env.task.get_gapartnet_anno()
            env.task.cal_handle(bbox_id=-1)
            policy.reset()
            franka_base_pos = env.task.franka_reset_pos_list[0]
            cabinet_pos = env.task.cabinet_reset_pos_tensor.cpu().numpy().tolist()

            if self.affordance:

                prompt = "cabinet"
                proj = env.task.cam_projs[0][0]
                view = env.task.cam_views[0][0]
                camera_w = env.task.cam_w
                camera_h = env.task.cam_h
                points_envs, colors_envs, masks_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, ori_masks_envs = env.task.get_camera_state()
                
                tgt_point_cloud = points_envs[0]
                tgt_point_color = colors_envs[0]
                tgt_point = np.concatenate((tgt_point_cloud, tgt_point_color), axis=1)
                tgt_mask = masks_envs[0]
                init_scene_img = Image.fromarray(rgb_envs[0][0]).convert('RGB')
                init_depth = depth_envs[0][0].transpose(1,0)

                retrieval_id = self.memory_buffer.retrieval_id(rgb_envs[0][0], self.task_name)
                retrieval_id_list.append(retrieval_id)
                retrieval_img, retrieval_afford, retrieval_point, retrieval_mask = self.memory_buffer.get_retrieval_info(retrieval_id, self.retrieval_path)
                contact_point = retrieval_afford[0]
                src_part, src_, src_seg_mask, src_shift, src_scale = run_pointsam(retrieval_point, retrieval_mask, contact_point, 40000)

                if contact_point.ndim != 2:
                    contact_point = contact_point.reshape(-1, 3)
                image_pixel = get_image_pixel_from_3d(contact_point, view, proj, camera_w, camera_h)
                retrieval_img.save('src.png')
                show_results(retrieval_img, image_pixel, save_name='src_afford')

                image_pixel = scale_img_pixel(image_pixel, camera_w, camera_h)
                mask_src_array, mask_src_img = run_sam(retrieval_img, prompt)
                mask_tgt_array, mask_tgt_img = run_sam(init_scene_img, prompt)
                tgt_pixel = transfer_pixel(mask_src_img, mask_tgt_img, prompt, image_pixel, ftype='sd_dinov2')
                tgt_pixel = scale_img_pixel(tgt_pixel, IMG_SIZE, IMG_SIZE, new_w=camera_w, new_h=camera_h)

                if mask_tgt_array[tgt_pixel[:,1], tgt_pixel[:,0]] == False:
                    tgt_pixel = find_nearest_object_pixel_in_box(mask_tgt_array, tgt_pixel)
                init_scene_img.save('tgt.png')
                show_results(init_scene_img, tgt_pixel, save_name='tgt_afford')
                
                tgt_pixel_depth = init_depth[tgt_pixel[:,0], tgt_pixel[:,1]]
                tgt_pixel_depth = np.array([tgt_pixel_depth]*tgt_pixel.shape[0])
                transfer_point = get_3d_from_image_pixel(tgt_pixel, tgt_pixel_depth, view, proj, camera_w, camera_h)
                tgt_part, tgt_, tgt_seg_mask, tgt_shift, tgt_scale = run_pointsam(tgt_point, tgt_mask, transfer_point, 40000)

                T, R, t = ICP_register(src_part, tgt_part, voxel_size=0.025, init="")

                transfer_point = (transfer_point-tgt_shift)/tgt_scale
                retrieval_afford[:, :3] = (retrieval_afford[:, :3]-src_shift)/src_scale

                transfer_afford = update_afford(transfer_point, retrieval_afford, R)
                transfer_afford = transfer_afford*tgt_scale + tgt_shift

                # vis

                obj_pc = tgt_point[tgt_mask[:, 0]==4]
                # obj_pc[:, :3] = (obj_pc[:, :3]-tgt_shift)/tgt_scale

                transfer_afford_color = np.zeros_like(transfer_afford)
                transfer_afford_color[:,0] = 255
                transfer_pc = np.concatenate((transfer_afford, transfer_afford_color), axis=1)

                # vis_pc = np.concatenate((obj_pc, transfer_pc), axis=0)
                vis_point_cloud(obj_pc, name='tgt_pc')
                src_pc = retrieval_point[retrieval_mask[:,0]==4]
                vis_point_cloud(src_pc, name='src_pc')
                # vis_point_cloud(retrieval_point[retrieval_mask==4])

                whole1 = src_ @ R.T + (transfer_point-retrieval_afford[0]).reshape(1, 3)
                whole = np.concatenate([whole1, tgt_], axis=0)
                value = np.concatenate(((src_seg_mask*1), (tgt_seg_mask*2)))

                draw(whole, "together", value)

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


                if self.dynamic_affordance:
                    EE = policy.forward_kinematics_unnorm(obs_dict_input['state'][0][:,-9:], franka_base_pos)
                    dsg_distance = torch.norm((EE.mean(0).cpu()-torch.from_numpy(transfer_afford[0]))).cpu().numpy()
                    print(dsg_distance)
                    if dsg_distance < policy.DSG_stop_threshold: # once grasped, stop dsg
                        policy.DSG_threshold = -0.1
                    # afford = env.task.get_dynamic_afford()
                    print(dsg_distance < policy.DSG_threshold)
                    print("\n")
                    # print(step_id)
                    afford = transfer_afford
                    afford = torch.from_numpy(afford.reshape(1,*afford.shape)).to(device=device)
                    # afford = torch.from_numpy(env.task.init_position.reshape(1,-1)).to(device=device)
                    afford = afford.repeat(self.n_obs_steps,1,1)
                    if dsg_distance < policy.DSG_threshold:
                        action_dict = policy.predict_action(obs_dict_input, afford, base_pose=franka_base_pos, use_dsg=True)
                    else:
                        action_dict = policy.predict_action(obs_dict_input, afford, base_pose=franka_base_pos)

                elif self.affordance:

                    afford = transfer_afford[0]
                    afford = torch.from_numpy(afford.reshape(1,*afford.shape)).to(device=device)
                    afford = afford.repeat(self.n_obs_steps,1,1)
                    action_dict = policy.predict_action(obs_dict_input, afford, franka_base_pos)

                else:
                    action_dict = policy.predict_action(obs_dict_input)

                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # path = os.path.join(save_dir,str(step_id)+".png")
                obs, success = env.step(action, save_dir, step_id)
                # env.task.record_frames(path)
                
            all_success_return.append(success.item())

            if success:
                success_pos.append(cabinet_pos)
        
        success_rate = all_success_return.count(1)/self.episode_eval
        print(f"---------------- Eval Results --------------")
        print("success rate:",success_rate)
        print(success_pos)
        np.savetxt("success_pos.txt", success_pos)

        if self.affordance:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("retrieval_id:", retrieval_id_list)



