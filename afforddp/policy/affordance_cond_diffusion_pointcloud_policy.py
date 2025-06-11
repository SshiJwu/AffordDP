from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import pytorch_kinematics as pk

from afforddp.model.common.normalizer import LinearNormalizer
from afforddp.policy.base_pointcloud_policy import BasePointcloudPolicy
from afforddp.model.diffusion.conditional_unet1d import ConditionalUnet1D
from afforddp.model.diffusion.mask_generator import LowdimMaskGenerator, PointcloudMaskGenerator
import afforddp.model.vision.crop_randomizer as dmvc
from afforddp.common.pytorch_util import dict_apply, replace_submodules
from afforddp.common.model_util import print_params
from afforddp.model.vision_3d.pointnet_extractor import DynamicAffordDPEncoder
from afforddp.model.vision_3d.se3_aug import create_se3_augmentation

def quat_pos_from_transform3d(tg):
    m = tg.get_matrix()
    pos = m[:, :3, 3]
    rot = pk.matrix_to_quaternion(m[:, :3, :3])
    return pos, rot

class AffordCondPointCloudPolicy(BasePointcloudPolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            DSG_threshold=0.15,
            DSG_stop_threshold=0.05,
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            afford_mlp_size=(64, 64),
            affordance_encoder_size=64,
            affordance_max_seq_len=6,
            num_attention_heads=2,
            num_attention_layers=2,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            se3_augmentation_cfg=None,
            pointcloud_encoder_cfg=None,
            # parameters passed to step
            **kwargs):
        super().__init__()

        self.condition_type = condition_type
        self.DSG_threshold = DSG_threshold
        self.DSG_stop_threshold = DSG_stop_threshold
        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        afford_shape = shape_meta['afford']['shape']


        obs_encoder = DynamicAffordDPEncoder(observation_space=obs_dict,
                                            afford_space=afford_shape,
                                            img_crop_shape=crop_shape,
                                            out_channel=encoder_output_dim,
                                            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                            use_pc_color=use_pc_color,
                                            pointnet_type=pointnet_type,
                                            afford_mlp_size=afford_mlp_size,
                                            affordance_encoder_size=affordance_encoder_size,
                                            max_seq_len=affordance_max_seq_len,
                                            num_attention_heads=num_attention_heads,
                                            num_attention_layers=num_attention_layers)

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")
        self.se3_augmentation_cfg = se3_augmentation_cfg
        if self.se3_augmentation_cfg.use_aug:
            self.se3_aug = create_se3_augmentation(self.se3_augmentation_cfg)
        else:
            self.se3_aug = None
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_aug: {self.se3_augmentation_cfg.use_aug}", "yellow")


        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps


        print_params(self)
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            afford=None,
            base_pose=None,
            use_dsg=False,
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler


        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)


        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]


            model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond)
            
            # 3. compute previous image: x_t -> x_t-1
            trajectory = trajectory.requires_grad_(True)

            if isinstance(scheduler, DDPMScheduler):
                scheduler_out = scheduler.step(
                    model_output, t, trajectory)
            else:
                scheduler_out = scheduler.step(
                    model_output, t, trajectory, eta=1)
                
            if use_dsg:

                contact_point = afford[:,:,0,:]
                # afford = (1,2,8,3)
                prev_sample = scheduler_out.prev_sample   #x-
                sigma_t = scheduler_out.std_dev_t
                prev_sample_mean = scheduler_out.prev_sample_mean
                pred_original_sample = scheduler_out.pred_original_sample

                trajectory, _ = self.DSG_conditioning(x_prev=trajectory,
                                                      x_t=prev_sample,
                                                      x_t_mean=prev_sample_mean,
                                                      x_0_hat=pred_original_sample,
                                                      sigma_t=sigma_t,
                                                      afford=contact_point,
                                                      base_pose=base_pose,
                                                      idx=t)
                # print("DSG_Output:", trajectory)
            else:
                trajectory = scheduler_out.prev_sample
            
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   

        
        return trajectory
    
    def DSG_conditioning(self, x_prev, x_t, x_t_mean, x_0_hat, afford, sigma_t, idx, interval=1, guidance_scale=0.5, base_pose=None):
        
        eps = 1e-8
        b, T, D = x_t.shape
        T_a = afford.shape[1]
        if idx%interval == 0:
            device = afford.device
            pos = self.forward_kinematics(x_0_hat, base_pose)
            pos = pos.view(b,T,3)
            afford = afford.repeat(1,int(T/T_a),1)
            difference = afford - pos.to(device)
            norm = torch.linalg.norm(difference).requires_grad_(True)
            # print(norm)
            # norm = torch.linalg.norm(difference,dim=[1,2]).requires_grad_(True)

            grad = torch.autograd.grad(outputs=norm, inputs=x_prev)
            grad_norm = torch.linalg.norm(grad[0])

            r = torch.sqrt(torch.tensor(D)) * sigma_t
            guidance_rate = guidance_scale

            d_star = -r * grad[0] / (grad_norm + eps)
            d_sample = x_t - x_t_mean

            mix_direction = d_sample + guidance_rate * (d_star - d_sample)
            mix_direction_norm = torch.linalg.norm(mix_direction, dim=[2])
            mix_step = mix_direction / (mix_direction_norm.unsqueeze(2).repeat(1,1,9) + eps) * r


            return x_t_mean + mix_step, norm
    


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], afford, base_pose=None, use_dsg=False) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        if afford.shape[:2] != value.shape[:2]:
            afford = afford.reshape(*value.shape[:2],-1, 3)
        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            condition_afford = afford
            afford = afford[:,:To,...].reshape(-1,*afford.shape[2:])
            nobs_features = self.obs_encoder(this_nobs,afford)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            afford=condition_afford,
            base_pose=base_pose,
            use_dsg=use_dsg,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction


        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        # print("pc before min, max, mean:", batch['obs']['point_cloud'][...,:3].max(), batch['obs']['point_cloud'][...,:3].min(), batch['obs']['point_cloud'][...,:3].mean())
        # print("image before min, max, mean:", batch['obs']['image'].max(), batch['obs']['image'].min(), batch['obs']['image'].mean())
        nobs = self.normalizer.normalize(batch['obs'])
        # print("pc after min, max, mean:", nobs['point_cloud'][...,:3].max(), nobs['point_cloud'][...,:3].min(), nobs['point_cloud'][...,:3].mean())
        # print("image after min, max, mean:", nobs['image'].max(), nobs['image'].min(), nobs['image'].mean())
        
        # print("action before, min, max, mean:", batch['action'].min(), batch['action'].max(), batch['action'].mean())
        nactions = self.normalizer['action'].normalize(batch['action'])
        afford = batch['afford']
        # print("action after, min, max, mean:", nactions.min(), nactions.max(), nactions.mean())

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        
        if self.se3_aug is not None:
            B, T, N, D = nobs['point_cloud'].shape
            nobs['point_cloud'] = self.se3_aug(nobs['point_cloud'].reshape(B*T, N, D)).reshape(B, T, N, D)

        
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            afford = afford[:,:self.n_obs_steps,...].reshape(-1,*afford.shape[2:])
            nobs_features = self.obs_encoder(this_nobs, afford)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()


        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        


        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        
        pred = self.model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                            local_cond=local_cond, 
                            global_cond=global_cond)


        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        

        loss_dict = {
                'bc_loss': loss.item(),
            }

        # print(f"t2-t1: {t2-t1:.3f}")
        # print(f"t3-t2: {t3-t2:.3f}")
        # print(f"t4-t3: {t4-t3:.3f}")
        # print(f"t5-t4: {t5-t4:.3f}")
        # print(f"t6-t5: {t6-t5:.3f}")
        
        return loss, loss_dict


    def forward_kinematics(self, x_t, base_pos):
        
        d = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32

        x_t = x_t.squeeze(0).to(dtype=dtype, device=d).requires_grad_(True)
        x_t = self.normalizer['action'].unnormalize(x_t)
        chain = pk.build_serial_chain_from_urdf(open("/home/user/Downloads/GAPartNet/manipulation/assets/urdf/franka_description/robots/franka_panda.urdf").read(),"panda_hand")
        chain = chain.to(dtype=dtype, device=d)

        ret = chain.forward_kinematics(x_t,end_only=False)
        leftfinger_tg = ret['panda_leftfinger']
        leftfinger_pos, _ = quat_pos_from_transform3d(leftfinger_tg)

        rightfinger_tg = ret['panda_rightfinger']
        rightfinger_pos, _ = quat_pos_from_transform3d(rightfinger_tg)

        pos = (leftfinger_pos+leftfinger_pos)/2
        pos = pos + torch.from_numpy(base_pos).to(dtype=dtype, device=d).reshape(1,-1)

        return pos
    
    def forward_kinematics_unnorm(self, x_t, base_pos):
        
        d = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32
        x_t = x_t.to(dtype)
        chain = pk.build_serial_chain_from_urdf(open("/home/user/Downloads/GAPartNet/manipulation/assets/urdf/franka_description/robots/franka_panda.urdf").read(),"panda_hand")
        chain = chain.to(dtype=dtype, device=d)

        ret = chain.forward_kinematics(x_t,end_only=False)
        leftfinger_tg = ret['panda_leftfinger']
        leftfinger_pos, _ = quat_pos_from_transform3d(leftfinger_tg)

        rightfinger_tg = ret['panda_rightfinger']
        rightfinger_pos, _ = quat_pos_from_transform3d(rightfinger_tg)

        pos = (rightfinger_pos+leftfinger_pos)/2
        pos = pos + torch.from_numpy(base_pos).to(dtype=dtype, device=d).reshape(1,-1)

        return pos

