from isaacgym import gymapi
# from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *
import imageio
import open3d as o3d
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import copy
import time
import trimesh as tm
from afforddp.gym_util.utils import  orientation_error, get_point_cloud_from_rgbd_GPU
import os, json
import yaml
from scipy.spatial.transform import Rotation as R
import sys

import plotly.graph_objects as go
import os
from pytorch3d.transforms import matrix_to_quaternion, quaternion_invert
import fpsample

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
    )
from curobo.geom.types import Mesh, WorldConfig, Cuboid
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.util_file import get_robot_path, join_path, load_yaml

def write_json(dict, file):

    with open(file, 'w', encoding='utf-8') as file:
        json.dump(dict, file, ensure_ascii=False, indent=4)

def quaternion2euler(quaternion):
    r = R.from_quat(quaternion)
    euler = r.as_euler('xyz', degrees=True)
    return euler


class CabinetManipEnv():
    def __init__(
            self, 
            cfgs,
            save_root = None
        ):
        self.cfgs = cfgs
        self.use_cam = cfgs["cam"]["use_cam"]
        self.asset_root = self.cfgs["asset"]["asset_root"]
        self.save_root = save_root
        self.headless = cfgs["HEADLESS"]
        self.use_gpu_pipeline = cfgs['sim']['use_gpu_pipeline']
        self.sim_device = cfgs['sim']['sim_device']
        self.compute_device_id = cfgs['sim']['compute_device_id']
        self.graphics_device_id = cfgs['sim']['graphics_device_id']
        physics_engine = cfgs['sim']['physics_engine']

        self.device = self.sim_device if self.use_gpu_pipeline else 'cpu'
        # if self.headless == True:
        #     self.graphics_device_id = -1
        if physics_engine == 'physx':
            self.physics_engine = gymapi.SIM_PHYSX
        else:
            self.physics_engine = gymapi.SIM_FLEX

        # configure env grid
        self.num_envs = cfgs["num_envs"]
        self.num_states = cfgs["num_states"]
        self.num_obs = cfgs["num_obs"]+cfgs["num_states"]
        self.num_actions = cfgs["num_actions"]
        self.num_per_row = int(math.sqrt(self.num_envs))
        self.spacing = cfgs["env_spacing"]
        self.env_lower = gymapi.Vec3(-self.spacing, -self.spacing, 0.0)
        self.env_upper = gymapi.Vec3(self.spacing, self.spacing, self.spacing)
        print("Creating %d environments" % self.num_envs)
        
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs, 7), device=self.device, dtype=torch.float)

        if self.use_cam:
            self.cam_w = cfgs["cam"]["cam_w"]
            self.cam_h = cfgs["cam"]["cam_h"]
            self.cam_far_plane = cfgs["cam"]["cam_far_plane"]
            self.cam_near_plane = cfgs["cam"]["cam_near_plane"]
            self.horizontal_fov = cfgs["cam"]["cam_horizontal_fov"]
            self.cam_poss = cfgs["cam"]["cam_poss"]
            self.cam_targets = cfgs["cam"]["cam_targets"]
            self.num_cam_per_env = len(self.cam_poss)
            self.point_cloud_bound = cfgs["cam"]["point_cloud_bound"]
             
        # acquire gym interface
        self.gym = gymapi.acquire_gym()
        
        # Grab controller
        self.controller_name = cfgs["controller"]
        assert self.controller_name in {"ik", "osc", "curobo"}, f"Invalid controller specified -- options are (ik, osc). Got: {self.controller_name}"
        
        self.create_sim()

        if not self.headless:
            self.create_viewer()
        
        self.add_ground_plane()

        self.use_robot = False
        self.use_rigid = False
        self.use_arti = False
        if self.cfgs["asset"]["robot"] is not None:
            self.use_robot = True
            self.prepare_franka_asset()

        if self.cfgs['asset']['rigid'] is not None:
            self.use_rigid = True
            self.prepare_obj_assets()

        if self.cfgs['asset']['arti'] is not None:
            self.use_arti = True
            self.prepare_arti_obj_assets()
        self.prepare_table()

        self.load_env(load_cam=self.use_cam)
        self.init_observation()
        self.num_dofs = self.dof_states.shape[0]
        self.init_dof_states = self.dof_states.view(self.num_envs, -1, 2).clone()
        self.init_dof_states[:,:self.franka_num_dofs,0] = torch.tensor([1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.04, 0.04],
                                                                       dtype=torch.float32).to('cuda')
        
        self.init_dof_states[:,self.franka_num_dofs:,0] = torch.tensor([0]*(self.num_dofs-self.franka_num_dofs),
                                                                       dtype=torch.float32).to('cuda')
        self.init_dof_states[:,self.franka_num_dofs:,1] = torch.tensor([0]*(self.num_dofs-self.franka_num_dofs),
                                                                       dtype=torch.float32).to('cuda')
        self.refresh_observation()

        if self.cfgs["USE_CUROBO"]:
            self.prepare_curobo(use_mesh=self.cfgs["USE_MESH_COLLISION"])

        self.enable_viewer_sync = True

        self.pos_dist_bar = cfgs["pos_dist_bar"]
        self.rot_dist_bar = cfgs["rot_dist_bar"]

        self.success = torch.zeros((self.num_envs,), device=self.device)
        self.success_buf = torch.zeros((self.num_envs,), device=self.device).long()
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        cabinet_pos_low = cfgs['asset']['arti']['arti_pos_low']
        cabinet_pos_upper = cfgs['asset']['arti']['arti_pos_upper']
        self.cabinet_pos_low = torch.tensor(cabinet_pos_low).to(self.device)  
        self.cabinet_pos_upper = torch.tensor(cabinet_pos_upper).to(self.device)
        
        franka_pos_low = cfgs['asset']['robot']['franka_pos_low']
        franka_pos_upper = cfgs['asset']['robot']['franka_pos_upper']
        self.franka_pos_low = torch.tensor(franka_pos_low).to(self.device)
        self.franka_pos_upper = torch.tensor(franka_pos_upper).to(self.device)

        self.cabinet_reset_pos_tensor = None

    def add_ground_plane(self):

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

    def create_sim(self):

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.use_gpu_pipeline = self.use_gpu_pipeline
        
        # unused for now, be careful, otherwise it will cause error
        sim_params.dt = 1.0 / 120.0 
        # sim_params.substeps = 2 # default 2
        # sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024 # default 1024 * 1024
        # sim_params.physx.max_depenetration_velocity = 1000
        # sim_params.physx.solver_type = 1
        # sim_params.physx.num_position_iterations = 8
        # sim_params.physx.num_velocity_iterations = 1
        # sim_params.physx.rest_offset = 0.0
        # sim_params.physx.contact_offset = 0.02
        # sim_params.physx.friction_offset_threshold = 0.001
        # sim_params.physx.friction_correlation_distance = 0.0005
        # sim_params.physx.num_threads = self.args.num_threads
        # sim_params.physx.use_gpu = self.args.use_gpu

        # create sim
        self.sim = self.gym.create_sim(self.compute_device_id, self.graphics_device_id, self.physics_engine, sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")
        
    def create_viewer(self):

        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")
            
    # used when controlling with curobo
    def prepare_curobo(self, use_mesh = False):
        setup_curobo_logger("error")
        tensor_args = TensorDeviceType()
        world_file = "collision_empty.yml"
        robot_file = "franka.yml"
        
        if not use_mesh:
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                robot_file,
                world_file,
                tensor_args,
                interpolation_dt=0.01,
            )
        else:

            if self.use_rigid:
                asset_files = self.cfgs["asset"]["asset_files"]
                asset_obj_files = [os.path.join(self.cfgs["asset"]["asset_root"],  "/".join(asset_file.split("/")[:-1]), "textured.obj") for asset_file in self.cfgs["asset"]["asset_files"]]
                # import pdb; pdb.set_trace()
                object_meshes = [tm.load(asset_obj_file) for asset_obj_file in asset_obj_files]
                states = self.root_states[2:, :7].cpu().numpy()
                # assert len(states) == len(object_meshes), "BUG!"
                obstables = [
                    Mesh(
                        name=f'object_{object_meshes_i}', 
                        pose=states[object_meshes_i],
                        vertices=object_meshes[object_meshes_i].vertices,
                        faces=object_meshes[object_meshes_i].faces
                        ) 
                    for object_meshes_i in range(len(object_meshes))
                    ]
                
                # import pdb; pdb.set_trace()
                table = Cuboid(
                    name='table',
                    dims=[self.table_scale[0], self.table_scale[1], self.table_scale[2]],
                    # dims=[0, 0, 0],
                    pose=[self.table_pose.p.x, self.table_pose.p.y, self.table_pose.p.z, self.table_pose.r.x, self.table_pose.r.y, self.table_pose.r.z, self.table_pose.r.w],
                    scale=1.0
                )
                world_model = WorldConfig(
                    mesh=obstables,
                    cuboid=[table],
                )

            else:

                table = Cuboid(
                    name='table',
                    dims=[self.table_scale[0], self.table_scale[1], self.table_scale[2]],
                    # dims=[0, 0, 0],
                    pose=[self.table_pose.p.x, self.table_pose.p.y, self.table_pose.p.z, self.table_pose.r.x, self.table_pose.r.y, self.table_pose.r.z, self.table_pose.r.w],
                    scale=1.0
                )
                world_model = WorldConfig(
                    cuboid=[table]
                )

            world_model = WorldConfig.create_collision_support_world(world_model)
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                robot_file,
                world_model,
                tensor_args,
                # interpolation_dt=0.1,
                # trajopt_tsteps=8,
                collision_checker_type=CollisionCheckerType.MESH,
                use_cuda_graph=False,
                # num_trajopt_seeds=12,
                # num_graph_seeds=12,
                # interpolation_dt=0.03,
                collision_cache={"obb": 30, "mesh": 10},
                # collision_activation_distance=0.01,
                # acceleration_scale=1.0,
                self_collision_check=True,
                maximum_trajectory_dt=0.25,
                # fixed_iters_trajopt=True,
                # finetune_dt_scale=1.05,
                # velocity_scale=None,
                # interpolation_type=InterpolateType.CUBIC,
                # use_gradient_descent=True,
                store_debug_in_result=False,
            )
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup(enable_graph=True)
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
        robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
       

    def get_gapartnet_anno(self):
        '''
        Get gapartnet annotation
        '''
        self.gapart_cates = []
        self.gapart_init_bboxes = []
        self.gapart_link_names = []
        self.gapart_raw_valid_annos = []
        for gapartnet_id in self.gapartnet_ids:
            # load object annotation
            annotation_path = f"{self.asset_root}/{self.gapartnet_root}/{gapartnet_id}/link_annotation_gapartnet.json"
            anno = json.loads(open(annotation_path).read())
            num_link_anno = len(anno)
            gapart_raw_valid_anno = []
            for link_i in range(num_link_anno):
                anno_i = anno[link_i]
                if anno_i["is_gapart"]:
                    gapart_raw_valid_anno.append(anno_i)
            self.gapart_raw_valid_annos.append(gapart_raw_valid_anno)
            self.gapart_cates.append([anno_i["category"] for anno_i in gapart_raw_valid_anno])
            self.gapart_init_bboxes.append(np.array([np.asarray(anno_i["bbox"]) for anno_i in gapart_raw_valid_anno]))
            self.gapart_link_names.append([anno_i["link_name"] for anno_i in gapart_raw_valid_anno])
        # print()

    def prepare_franka_asset(self):
        '''
        Prepare franka asset
        '''
        # load franka asset
        franka_asset_file = self.cfgs["asset"]["robot"]["franka_asset_file"]
        self.franka_seg_id = self.cfgs["asset"]["robot"]["franka_seg_id"]
        self.franka_reset_dof_pos_interval = self.cfgs["asset"]["robot"]["franka_reset_dof_pos_interval"]

        arti_obj_asset_options = gymapi.AssetOptions()
        arti_obj_asset_options.armature = 0.1
        arti_obj_asset_options.fix_base_link = True
        arti_obj_asset_options.disable_gravity = True
        # Switch Meshes from Z-up left-handed system to Y-up Right-handed coordinate system.
        arti_obj_asset_options.flip_visual_attachments = True
        arti_obj_asset_options.collapse_fixed_joints = False 
        arti_obj_asset_options.thickness = 0.5 # default 0.02
        arti_obj_asset_options.vhacd_enabled = True
        arti_obj_asset_options.vhacd_params = gymapi.VhacdParams()
        # arti_obj_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        arti_obj_asset_options.vhacd_params.resolution = 512 # 1000000
        arti_obj_asset_options.use_mesh_materials = True
        self.franka_asset = self.gym.load_asset(self.sim, self.asset_root, franka_asset_file, arti_obj_asset_options)

        # configure franka dofs
        self.franka_dof_props = self.gym.get_asset_dof_properties(self.franka_asset)
        franka_lower_limits = self.franka_dof_props["lower"]
        franka_upper_limits = self.franka_dof_props["upper"]
        self.franka_lower_limit_tensor = torch.tensor(franka_lower_limits).cuda()
        self.franka_upper_limit_tensor = torch.tensor(franka_upper_limits).cuda()
        
        # Set controller parameters
        # use position drive for all dofs
        if self.controller_name == "ik" or self.controller_name == "curobo":
            self.franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
            self.franka_dof_props["stiffness"][:7].fill(400.0)
            self.franka_dof_props["damping"][:7].fill(40.0)
        else:       # osc
            self.franka_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
            self.franka_dof_props["stiffness"][:7].fill(0.0)
            self.franka_dof_props["damping"][:7].fill(0.0)
        # grippers
        self.franka_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
        self.franka_dof_props["stiffness"][7:].fill(1.0e6)
        self.franka_dof_props["damping"][7:].fill(100.0)
        
        # default dof states and position targets
        self.franka_num_dofs = self.gym.get_asset_dof_count(self.franka_asset)
        self.franka_default_dof_pos = np.zeros(self.franka_num_dofs, dtype=np.float32)
        # self.franka_default_dof_pos[:7] = franka_mids[:7] 
        self.franka_default_dof_pos[:7] = np.array(self.cfgs["asset"]["robot"]["franka_init_dof"], dtype=np.float32)
        # grippers open
        self.franka_default_dof_pos[7:] = franka_upper_limits[7:]

        self.franka_default_dof_state = np.zeros(self.franka_num_dofs, gymapi.DofState.dtype)
        self.franka_default_dof_state["pos"] = self.franka_default_dof_pos

        # send to torch
        self.default_dof_pos_tensor = to_torch(self.franka_default_dof_pos, device=self.device)

        # get link index of panda hand, which we will use as end effector
        franka_link_dict = self.gym.get_asset_rigid_body_dict(self.franka_asset)
        self.franka_num_links = len(franka_link_dict)
        # print("franka dof:", self.franka_num_dofs, "franka links:", self.franka_num_links)
        self.franka_hand_index = franka_link_dict["panda_hand"]

    def prepare_table(self):
        
        arti_obj_asset_options = gymapi.AssetOptions()
        arti_obj_asset_options.fix_base_link = True
       
        table_pose_p = self.cfgs["asset"]["table"]["table_pose_p"]
        table_scale = self.cfgs["asset"]["table"]["table_scale"]
        self.table_scale = self.cfgs["asset"]["table"]["table_scale"]
        self.table_seg_id = self.cfgs["asset"]["table"]["table_seg_id"]
        table_dims = gymapi.Vec3(table_scale[0], table_scale[1], table_scale[2])
        self.table_pose = gymapi.Transform()
        self.table_pose.p = gymapi.Vec3(table_pose_p[0], table_pose_p[1], table_pose_p[2])
        self.table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, arti_obj_asset_options)


    def prepare_obj_assets(self):
        '''
        Prepare object assets, some ycb or objaverse objects
        '''

        arti_obj_asset_options = gymapi.AssetOptions()
        arti_obj_asset_options.fix_base_link = True

        obj_asset_files = self.cfgs["asset"]["rigid"]["asset_files"]
        self.rigid_seg_id = self.cfgs["asset"]["arti"]["rigid_seg_id"]
        arti_obj_asset_options = gymapi.AssetOptions()
        arti_obj_asset_options.use_mesh_materials = True
        arti_obj_asset_options.disable_gravity = True
        arti_obj_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        arti_obj_asset_options.override_inertia = True
        arti_obj_asset_options.override_com = True
        arti_obj_asset_options.vhacd_enabled = True
        arti_obj_asset_options.vhacd_params = gymapi.VhacdParams()
        arti_obj_asset_options.vhacd_params.resolution = 1000000
        self.num_asset_per_env = len(obj_asset_files)
        self.obj_assets = [self.gym.load_asset(self.sim, self.asset_root, obj_asset_file, arti_obj_asset_options) for obj_asset_file in obj_asset_files]
        self.obj_num_links_dict = [self.gym.get_asset_rigid_body_dict(asset_i) for asset_i in self.obj_assets]
        self.obj_num_links = sum([len(obj_num_links) for obj_num_links in self.obj_num_links_dict])
        self.obj_num_dofs = sum([self.gym.get_asset_dof_count(asset_i) for asset_i in self.obj_assets])
        self.table_num_links = 1
    

    def prepare_arti_obj_assets(self):
        '''
        Prepare articulated object assets
        '''
               ### TODO: support multiple loading
        self.gapartnet_ids = self.cfgs["asset"]["arti"]["arti_gapartnet_ids"]
        self.gapartnet_root = self.cfgs["asset"]["arti"]["arti_obj_root"]
        self.arti_seg_id = self.cfgs["asset"]["arti"]["arti_seg_id"]
        arti_obj_paths = [f"{self.gapartnet_root}/{gapartnet_id}/mobility_annotation_gapartnet.urdf" for gapartnet_id in self.gapartnet_ids]

        arti_obj_asset_options = gymapi.AssetOptions()
        arti_obj_asset_options.disable_gravity = True     # if not disabled, it will need a very initial large force to open a drawer
        arti_obj_asset_options.fix_base_link = True 
        arti_obj_asset_options.collapse_fixed_joints = False # default False
        # arti_obj_asset_options.convex_decomposition_from_submeshes = True
        # arti_obj_asset_options.armature = 0.005 # default 0.0
        arti_obj_asset_options.vhacd_enabled = True
        arti_obj_asset_options.vhacd_params = gymapi.VhacdParams()
        arti_obj_asset_options.vhacd_params.resolution = 512 # 1000000
        arti_obj_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        arti_obj_asset_options.flip_visual_attachments = False
        
        self.arti_obj_assets = [self.gym.load_asset(self.sim, self.asset_root, arti_obj_path, arti_obj_asset_options)
                                for arti_obj_path in arti_obj_paths]

        ### TODO: support multiple loading from here
        self.arti_obj_asset = self.arti_obj_assets[0]
        self.arti_obj_num_dofs = self.gym.get_asset_dof_count(self.arti_obj_asset)
        arti_obj_link_dict = self.gym.get_asset_rigid_body_dict(self.arti_obj_asset)
        self.arti_obj_num_links = len(arti_obj_link_dict)
        # print("obj dof:", self.arti_obj_num_dofs, "obj links:", self.arti_obj_num_links)
        
        # set physical props
        self.arti_obj_dof_props = self.gym.get_asset_dof_properties(self.arti_obj_asset)
        # self.arti_obj_dof_props['stiffness'][:] = 10.0 
        self.arti_obj_dof_props['damping'][:] = 10.0      # large damping can reduce interia(?)
        # self.arti_obj_dof_props['friction'][:] = 5.0
        self.arti_obj_dof_props["driveMode"][:] = gymapi.DOF_MODE_NONE
        
        init_pos = self.arti_obj_dof_props["lower"]
        self.arti_obj_default_dof_pos = np.zeros(self.arti_obj_num_dofs, dtype=np.float32)
        self.arti_obj_default_dof_state = np.zeros(self.arti_obj_num_dofs, gymapi.DofState.dtype)
        self.arti_obj_default_dof_state["pos"] = init_pos
        self.arti_default_dof_pos_tensor = to_torch(self.arti_obj_default_dof_pos, device=self.device)


    def load_env(self, load_cam = True):
        '''
        Load environment
        '''
        self.envs = []
        self.obj_actor_idxs = []
        self.hand_idxs = []
        self.init_franka_pos_list = []
        self.init_franka_rot_list = []
        self.init_obj_pos_list = []
        self.init_obj_rot_list = []
        self.arti_init_obj_pos_list = []
        self.arti_init_obj_rot_list = []
        self.env_offsets = []
        self.arti_obj_actor_idxs = []
        self.franka_handles = []
        self.arti_obj_handles = []

        if self.use_robot:
            franka_pose = gymapi.Transform()
            franka_pose_p = self.cfgs["asset"]["robot"]["franka_pose_p"]
            franka_pose.p = gymapi.Vec3(franka_pose_p[0], franka_pose_p[1], franka_pose_p[2])
            self.franka_reset_position_noise = self.cfgs["asset"]["robot"]["franka_reset_position_noise"]
            self.franka_reset_rotation_noise = self.cfgs["asset"]["robot"]["franka_reset_rotation_noise"]
        
        if self.use_rigid:
            obj_pose_ps = [self.cfgs["asset"]["rigid"]["obj_pose_ps"][obj_i] for obj_i in range(self.num_asset_per_env)]
            if self.cfgs["asset"]["rigid"]["obj_pose_rs"] is not None:
                obj_pose_rs = [self.cfgs["asset"]["rigid"]["obj_pose_rs"][obj_i] for obj_i in range(self.num_asset_per_env)]
            else:
                obj_pose_rs = None
            
            position_noise = self.cfgs["asset"]["rigid"]["position_noise"]
            rotation_noise = self.cfgs["asset"]["rigid"]["rotation_noise"]
        
        if self.use_arti:

            arti_obj_pose_ps = self.cfgs["asset"]["arti"]["arti_obj_pose_ps"]
            arti_obj_pose_p = arti_obj_pose_ps[0]   
            self.arti_reset_position_noise = self.cfgs["asset"]["arti"]["arti_reset_position_noise"]
            self.arti_reset_rotation_noise = self.cfgs["asset"]["arti"]["arti_reset_rotation_noise"]
            arti_rotation = self.cfgs["asset"]["arti"]["arti_rotation"]
        
 
        if load_cam:
            self.cams = []
            self.rgb_tensors = []
            self.depth_tensors = []
            self.seg_tensors = []
            self.cam_vinvs = []
            self.cam_projs = []
            self.cam_views = []

        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(self.sim, self.env_lower, self.env_upper, self.num_per_row)
            self.envs.append(env)
            origin = self.gym.get_env_origin(env)
            self.env_offsets.append([origin.x, origin.y, origin.z])

            if self.use_robot:

                franka_handle = self.gym.create_actor(env, self.franka_asset, franka_pose, "franka", i, 2, self.franka_seg_id) #self.franka_seg_id
                # set dof properties
                self.gym.set_actor_dof_properties(env, franka_handle, self.franka_dof_props)

                # set initial dof states
                
                self.gym.set_actor_dof_states(env, franka_handle, self.franka_default_dof_state, gymapi.STATE_ALL)

                # set initial position targets
                # print(self.franka_default_dof_pos)
                self.gym.set_actor_dof_position_targets(env, franka_handle, self.franka_default_dof_pos)

                # get inital hand pose
                hand_handle = self.gym.find_actor_rigid_body_handle(env, franka_handle, "panda_hand")
                hand_pose = self.gym.get_rigid_transform(env, hand_handle)
                self.init_franka_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
                self.init_franka_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

                # get global index of hand in rigid body state tensor
                hand_idx = self.gym.find_actor_rigid_body_index(env, franka_handle, "panda_hand", gymapi.DOMAIN_SIM)
                self.hand_idxs.append(hand_idx)

                franka_shape_props = self.gym.get_actor_rigid_shape_properties(env, franka_handle)
                for franka_shape_prop in franka_shape_props:

                    franka_shape_prop.contact_offset = 0.001 # 0.001

                    franka_shape_prop.friction = 5.0
                    franka_shape_prop.filter = 2

                self.gym.set_actor_rigid_shape_properties(env, franka_handle, franka_shape_props)  
                self.franka_handles.append(franka_handle)

            ### Table
            self.table_handle = self.gym.create_actor(env, self.table_asset, self.table_pose, "table", i, 0, self.table_seg_id)
            
            if self.use_rigid:

                self.init_obj_pos_list.append([])
                self.init_obj_rot_list.append([])
                self.obj_actor_idxs.append([])
                for asset_i in range(self.num_asset_per_env):
                    initial_pose = gymapi.Transform()
                    initial_pose.p.x = obj_pose_ps[asset_i][0] 
                    initial_pose.p.y = obj_pose_ps[asset_i][1] 
                    initial_pose.p.z = obj_pose_ps[asset_i][2]
                    if obj_pose_rs is None:
                        initial_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), rotation_noise/180.0*np.random.uniform(-math.pi, math.pi))
                    else:
                        initial_pose.r.x =  obj_pose_rs[asset_i][0]
                        initial_pose.r.y =  obj_pose_rs[asset_i][1]
                        initial_pose.r.z =  obj_pose_rs[asset_i][2]
                        initial_pose.r.w =  obj_pose_rs[asset_i][3]
                    # initial_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), rotation_noise/180.0*np.random.uniform(-math.pi, math.pi))

                    self.init_obj_pos_list[-1].append([initial_pose.p.x, initial_pose.p.y, initial_pose.p.z])
                    self.init_obj_rot_list[-1].append([initial_pose.r.x, initial_pose.r.y, initial_pose.r.z, initial_pose.r.w])
                    
                    obj_actor_handle = self.gym.create_actor(env, self.obj_assets[asset_i], initial_pose, f'actor_{asset_i}', i, 0, self.asset_seg_ids[asset_i])
                    
                    obj_actor_idx = self.gym.get_actor_rigid_body_index(env, obj_actor_handle, 0, gymapi.DOMAIN_SIM)
                    self.obj_actor_idxs[i].append(obj_actor_idx)
                    self.gym.set_actor_scale(env, obj_actor_handle, self.cfgs["asset"]["rigid"]["obj_scale"])
            
            if self.use_arti:

                arti_initial_pose = gymapi.Transform()
                arti_initial_pose.p.x = arti_obj_pose_p[0] 
                arti_initial_pose.p.y = arti_obj_pose_p[1] 
                arti_initial_pose.p.z = arti_obj_pose_p[2]
                arti_initial_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), arti_rotation/180.0*math.pi)
                self.arti_init_obj_pos_list.append([arti_initial_pose.p.x, arti_initial_pose.p.y, arti_initial_pose.p.z])
                self.arti_init_obj_rot_list.append([arti_initial_pose.r.x, arti_initial_pose.r.y, arti_initial_pose.r.z, arti_initial_pose.r.w])
                arti_obj_actor_handle = self.gym.create_actor(env, self.arti_obj_asset, arti_initial_pose, 'arti_actor', i, 1, self.arti_seg_id) #1, self.asset_seg_ids[-1] + 1
                

                self.gym.set_actor_dof_properties(env, arti_obj_actor_handle, self.arti_obj_dof_props)
                # set initial dof states
                self.gym.set_actor_dof_states(env, arti_obj_actor_handle, self.arti_obj_default_dof_state, gymapi.STATE_ALL)
                # # set initial position targets
                self.gym.set_actor_dof_position_targets(env, arti_obj_actor_handle, self.arti_obj_default_dof_pos)
                arti_obj_actor_idx = self.gym.get_actor_rigid_body_index(env, arti_obj_actor_handle, 0, gymapi.DOMAIN_SIM)
                self.arti_obj_actor_idxs.append(arti_obj_actor_idx)
                self.gym.set_actor_scale(env, arti_obj_actor_handle, self.cfgs["asset"]["arti"]["arti_obj_scale"])
                
                agent_shape_props = self.gym.get_actor_rigid_shape_properties(env, arti_obj_actor_handle)
                for agent_shape_prop in agent_shape_props:
                    # agent_shape_prop.compliance = agent.rigid_shape_compliance
                    agent_shape_prop.contact_offset = 0.001 # 0.001
                    # agent_shape_prop.filter = agent.rigid_shape_filter
                    agent_shape_prop.friction = 5.0
                    agent_shape_prop.filter = 1
                    # agent_shape_prop.rest_offset = agent.rigid_shape_rest_offset
                    # agent_shape_prop.restitution = agent.rigid_shape_restitution
                    # agent_shape_prop.rolling_friction = agent.rigid_shape_rolling_friction
                    # agent_shape_prop.thickness = 1
                    # agent_shape_prop.torsion_friction = agent.rigid_shape_torsion_friction
                self.gym.set_actor_rigid_shape_properties(env, arti_obj_actor_handle, agent_shape_props)  
                self.arti_obj_handles.append(arti_obj_actor_handle)

            if load_cam:
                # add camera
                cam_props = gymapi.CameraProperties()
                cam_props.width = self.cam_w
                cam_props.height = self.cam_h
                cam_props.far_plane = self.cam_far_plane
                cam_props.near_plane = self.cam_near_plane 
                cam_props.horizontal_fov = self.horizontal_fov
                cam_props.enable_tensors = True
                self.cams.append([])
                self.depth_tensors.append([])
                self.rgb_tensors.append([])
                self.seg_tensors.append([])
                self.cam_vinvs.append([])
                self.cam_projs.append([])
                self.cam_views.append([])
                for i in range(self.num_cam_per_env):
                    cam_handle = self.gym.create_camera_sensor(env, cam_props)
                    self.gym.set_camera_location(cam_handle, env, 
                        gymapi.Vec3(self.cam_poss[i][0], self.cam_poss[i][1], self.cam_poss[i][2]), 
                        gymapi.Vec3(self.cam_targets[i][0], self.cam_targets[i][1], self.cam_targets[i][2]))
                    self.cams[-1].append(cam_handle)
                
                    proj = self.gym.get_camera_proj_matrix(self.sim, env, cam_handle)
                    view = self.gym.get_camera_view_matrix(self.sim, env, cam_handle)
                    # view_matrix_inv = torch.inverse(torch.tensor(self.gym.get_camera_view_matrix(self.sim, env, cam_handle))).to(self.device)
                    vinv = np.linalg.inv(np.matrix(view))
                    self.cam_vinvs[-1].append(vinv)
                    self.cam_projs[-1].append(proj)
                    self.cam_views[-1].append(view)

                    # obtain rgb tensor
                    rgb_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env, cam_handle, gymapi.IMAGE_COLOR)
                    # wrap camera tensor in a pytorch tensor
                    torch_rgb_tensor = gymtorch.wrap_tensor(rgb_tensor)
                    self.rgb_tensors[-1].append(torch_rgb_tensor)
                    
                    # obtain depth tensor
                    depth_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env, cam_handle, gymapi.IMAGE_DEPTH)
                    # wrap camera tensor in a pytorch tensor
                    torch_depth_tensor = gymtorch.wrap_tensor(depth_tensor)
                    self.depth_tensors[-1].append(torch_depth_tensor)
        

                    # obtain depth tensor
                    seg_tensor = self.gym.get_camera_image_gpu_tensor(
                        self.sim, env, cam_handle, gymapi.IMAGE_SEGMENTATION)
                    # wrap camera tensor in a pytorch tensor
                    torch_seg_tensor = gymtorch.wrap_tensor(seg_tensor)
                    self.seg_tensors[-1].append(torch_seg_tensor)
                        
        self.env_offsets = np.array(self.env_offsets)
        
        # point camera at middle env
        if not self.headless:
            
            viewer_cam_pos = gymapi.Vec3(self.cam_poss[0][0], self.cam_poss[0][1], self.cam_poss[0][2])
            viewer_cam_target = gymapi.Vec3(self.cam_targets[0][0], self.cam_targets[0][1], self.cam_targets[0][2])
            middle_env = self.envs[self.num_envs // 2 + self.num_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, viewer_cam_pos, viewer_cam_target)

        # from now on, we will use the tensor API that can run on CPU or GPU
        self.gym.prepare_sim(self.sim)

    def control_ik(self, dpose):
        # global damping, j_eef, num_envs
        damping = 0.05
        # solve damped least squares
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
        return u
    
    def plan_to_pose_ik(self, goal_position, goal_roation, close_gripper = True, save_video = False, save_root = "", start_step = 0, control_steps = 10):
        pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        effort_action = torch.zeros_like(pos_action)
        hand_rot_now = self.hand_rot
        goal_roation = goal_roation.to(self.device).reshape(1,-1)
        goal_position = goal_position.to(self.device).reshape(1,-1)
        if goal_roation.shape[1] != 0:
            orn_err = orientation_error(goal_roation, hand_rot_now)
        else:
            orn_err = hand_rot_now[...,:3].clone()
            orn_err[...] = 0
        pos_err = goal_position - self.hand_pos
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        pos_action[:, :7] = self.dof_pos.squeeze(-1)[:, :7] + self.control_ik(dpose)
        if close_gripper:
            grip_acts = torch.Tensor([[0., 0.]] * self.num_envs).to(self.device)
        else:
            grip_acts = torch.Tensor([[1, 1]] * self.num_envs).to(self.device)
        # grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * self.num_envs).to(self.device), 
        #                         torch.Tensor([[0.04, 0.04]] * self.num_envs).to(self.device))
        pos_action[:, 7:9] = grip_acts
        for step_i in range(control_steps):
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(effort_action))            
            self.run_steps(pre_steps = 1)
            if save_video:
                self.gym.render_all_camera_sensors(self.sim)
                step_str = str(start_step + step_i).zfill(4)
                os.makedirs(f"{save_root}/video", exist_ok=True)
                self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.cams[0][0], gymapi.IMAGE_COLOR, f"{save_root}/video/step-{step_str}.png")

    def init_observation(self):
        # get jacobian tensor
        # for fixed-base franka, tensor has shape (num envs, 10, 6, 9)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "franka")
        self.jacobian = gymtorch.wrap_tensor(_jacobian)

        # jacobian entries corresponding to franka hand
        self.j_eef = self.jacobian[:, self.franka_hand_index - 1, :, :7]

        # get mass matrix tensor
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "franka")
        self.mm = gymtorch.wrap_tensor(_massmatrix)
        self.mm = self.mm[:, :7, :7]          # only need elements corresponding to the franka arm
        
        # get rigid body state tensor
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)
        num_rb = int(self.rb_states.shape[0]/self.num_envs)
        
        self.root_states = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim))
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = self.root_states.view(self.num_envs, -1, 13)
            
        # get dof state tensor
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)
        # print("init:",self.dof_states)
        # self.gym.refresh_dof_state_tensor(self.sim)
        # print("refresh:",self.dof_states)
       
        num_dof = int(self.dof_states.shape[0]/self.num_envs)
    
        self.dof_pos = self.dof_states[:, 0].view(self.num_envs, num_dof, 1)
        self.dof_vel = self.dof_states[:, 1].view(self.num_envs, num_dof, 1)

    def _get_base_observations(self):

        points_envs, colors_envs, masks_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, ori_masks_envs = self.get_camera_state()

        point = points_envs[0][:,:3]
        color = colors_envs[0][:,:3]
        mask = masks_envs[0]

        samples_idx = fpsample.bucket_fps_kdline_sampling(point[:,:3], 4096,h=7)
        point = torch.from_numpy(point[samples_idx]).to(self.device)
        color = torch.from_numpy(color[samples_idx]).to(self.device)
        mask = torch.from_numpy(mask[samples_idx]).to(self.device)

        obs = torch.concatenate((point,color,mask),dim=1).repeat(self.num_envs,1,1)

        franka_dof_pose = self.dof_states[:self.franka_num_dofs,0]
        states = torch.concatenate((self.hand_pos,self.hand_rot,
                                   self.root_states[0,0,:7].view(1,-1),
                                   franka_dof_pose.view(1,-1)),dim=1).view(self.num_envs,-1,1).repeat(self.num_envs,1,7)
        
        self.obs_buf = torch.concatenate((obs,states),dim=1)

        return self.obs_buf


    def refresh_observation(self, get_visual_obs = True):
        # refresh tensors
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        
        # state obs
        self.hand_pos = self.rb_states[self.hand_idxs, :3]
        self.hand_rot = self.rb_states[self.hand_idxs, 3:7]
        self.hand_vel = self.rb_states[self.hand_idxs, 7:]

        ### TODO: support different dof tensor shapes in different envs
        self.robot_dof_qpos_qvel = self.dof_states.reshape(self.num_envs,-1,2)[:,:self.franka_num_dofs, :].view(self.num_envs, self.franka_num_dofs, 2)

        self.obs_buf = self._get_base_observations()

        return self.obs_buf
    
    def get_camera_state(self):
                # render sensors and refresh camera tensors
        if self.use_cam :

            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            points_envs = []
            colors_envs = []
            masks_envs = []
            ori_points_envs = []
            ori_colors_envs = []
            ori_masks_envs  = []
            rgb_envs = []
            depth_envs = []
            seg_envs = []
            for env_i in range(self.num_envs):
                points_env = []
                colors_env = []
                masks_env = []
                rgb_env = []
                depth_env = []
                seg_env = []
                for cam_i_per_env in range(self.num_cam_per_env):
                    # write tensor to image
                    cam_img = self.rgb_tensors[env_i][cam_i_per_env].cpu().numpy()
                    depth = self.depth_tensors[env_i][cam_i_per_env].cpu().numpy() # W * H
                    seg = self.seg_tensors[env_i][cam_i_per_env].cpu().numpy() # W * H

                    rgb_env.append(cam_img)
                    depth_env.append(depth)
                    seg_env.append(seg)
                    
                    ### RGBD -> Point Cloud with GPU
                    s = time.time()
                    pointclouds = get_point_cloud_from_rgbd_GPU(
                        self.depth_tensors[env_i][cam_i_per_env], 
                        self.rgb_tensors[env_i][cam_i_per_env],
                        self.seg_tensors[env_i][cam_i_per_env],
                        self.cam_vinvs[env_i][cam_i_per_env], 
                        self.cam_projs[env_i][cam_i_per_env], 
                        self.cam_w, self.cam_h
                    )
                    points = pointclouds[:, :3].cpu().numpy()
                    colors = pointclouds[:, 3:6].cpu().numpy()
                    masks = pointclouds[:,6:].cpu().numpy()

                    points_env.append(points)
                    colors_env.append(colors)
                    masks_env.append(masks)
                    
                ori_points_envs.append(points_env)
                ori_colors_envs.append(colors_env)
                ori_masks_envs.append(masks_env)

                rgb_envs.append(rgb_env)
                depth_envs.append(depth_env)
                seg_envs.append(seg_env)

                points_env = np.concatenate(points_env, axis=0) - self.env_offsets[env_i]
                colors_env = np.concatenate(colors_env, axis=0) - self.env_offsets[env_i]
                pc_mask_bound = (points_env[:, 0] > self.point_cloud_bound[0][0]) & (points_env[:, 0] < self.point_cloud_bound[0][1]) & \
                                (points_env[:, 1] > self.point_cloud_bound[1][0]) & (points_env[:, 1] < self.point_cloud_bound[1][1]) & \
                                (points_env[:, 2] > self.point_cloud_bound[2][0]) & (points_env[:, 2] < self.point_cloud_bound[2][1])
                points_env = points_env[pc_mask_bound]
                colors_env = colors_env[pc_mask_bound]
                masks_env = np.concatenate(masks_env, axis=0)
                masks_env = masks_env[pc_mask_bound]
                # s = time.time()
                # points_env, colors_env, pcs_mask = get_downsampled_pc(points_env, colors_env, 
                #     sampled_num=self.cfgs["cam"]["sampling_num"], sampling_method = self.cfgs["cam"]["sampling_method"])
                # e = time.time()
                points_envs.append(points_env)
                colors_envs.append(colors_env)
                masks_envs.append(masks_env)
                # print("Time to get point cloud: ", e-s)

            self.gym.end_access_image_tensors(self.sim)

        return points_envs, colors_envs, masks_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, ori_masks_envs

    def save_render(self, rgb_envs, depth_envs, ori_points_env, ori_colors_env, points, colors, save_dir, save_name = "render", save_pc = False, save_depth = False):
       
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        for env_i in range(len(rgb_envs)):
            for cam_i in range(len(rgb_envs[0])):
                fname = os.path.join(save_dir, f"{save_name}-rgb-{env_i}-{cam_i}.png")
                os.makedirs(save_dir, exist_ok=True)
                imageio.imwrite(fname, rgb_envs[env_i][cam_i].astype(np.uint8))

                if depth_envs is not None and save_depth:
                    depth = depth_envs[env_i][cam_i]
                    # depth clip to 0.1m - 10m and scale to 0-255
                    depth_clip = np.clip(depth, -1, 1)
                    depth_rgb = (depth_clip + 1) / 2 * 255.0
                    # W * H * 3
                    depth_img = np.zeros((depth_rgb.shape[0], depth_rgb.shape[1], 3))
                    depth_img[:, :, 0] = depth_rgb
                    depth_img[:, :, 1] = depth_rgb
                    depth_img[:, :, 2] = depth_rgb
                    fname = os.path.join(save_dir, f"{save_name}-depth-{env_i}-{cam_i}.png")
                    os.makedirs(save_dir, exist_ok=True)
                    imageio.imwrite(fname, depth_img.astype(np.uint8))
            
                if ori_points_env is not None and save_pc:
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(ori_points_env[env_i][cam_i][:, :3])
                    point_cloud.colors = o3d.utility.Vector3dVector(ori_colors_env[env_i][cam_i][:, :3]/255.0)
                    # save_to ply
                    fname = os.path.join(save_dir, f"{save_name}-partial-point_cloud--{env_i}-{cam_i}.ply")
                    o3d.io.write_point_cloud(fname, point_cloud)

            # o3d.visualization.draw_geometries([point_cloud])
            # if points is not None and save_pc:
            #     point_cloud = o3d.geometry.PointCloud()
            #     point_cloud.points = o3d.utility.Vector3dVector(points[env_i][:, :3])
            #     point_cloud.colors = o3d.utility.Vector3dVector(colors[env_i][:, :3]/255.0)
            #     # save_to ply
            #     fname = os.path.join(save_dir, f"{save_name}-{env_i}-all-point_cloud.ply")
            #     o3d.io.write_point_cloud(fname, point_cloud)

    def record_frames(self,path):
        
        self.gym.write_viewer_image_to_file(self.viewer,path)

    def plan_single_pose_curobo(self, position, quaternion, max_attempts=100, start_state= None):


        self.franka_pos = self.root_states[0,0,:3]
        if start_state == None:
            start_state = JointState.from_position(self.robot_dof_qpos_qvel[:,:7,0])
        goal_state = Pose(torch.tensor(torch.tensor(position)-self.franka_pos, device = self.device, dtype = torch.float64), 
                          quaternion=torch.tensor(quaternion, device = self.device, dtype = torch.float64))
        result = self.motion_gen.plan_single(start_state, goal_state, MotionGenPlanConfig(max_attempts=max_attempts))

        return result

    def plan_to_pose_curobo(self, position, quaternion, max_attempts=100, start_state= None):
        '''
        start_state: JointState
            if None, use current state as start state
            else, use given start_state
            
        position: list or np.array
            target position
        quaternion: list or np.array
            target orientation
        '''

        self.franka_pos = self.root_states[0,0,:3]
        if start_state == None:
            start_state = JointState.from_position(self.robot_dof_qpos_qvel[:,:7,0])
        goal_state = Pose(torch.tensor(torch.tensor(position)-self.franka_pos.cpu(), device = self.device, dtype = torch.float64), 
                          quaternion=torch.tensor(quaternion, device = self.device, dtype = torch.float64))
        result = self.motion_gen.plan_single(start_state, goal_state, MotionGenPlanConfig(max_attempts=max_attempts))

        traj = result.get_interpolated_plan()

        try:
            print("Trajectory Generated: ", result.success, result.optimized_dt.item(), traj.position.shape)
        except:
            print("Trajectory Generated: ", result.success)
        return traj

    def move_to_traj(self, traj, close_gripper = True, save_video = False, save_root = "", start_step = 0):
        pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        effort_action = torch.zeros_like(pos_action)
        #import pdb; pdb.set_trace()
        for step_i in range(len(traj)):
            # print("Step: ", step_i)
            # Deploy actions
            pos_action[:, :7] = traj.position.reshape(-1, 7)[step_i]
            if close_gripper:
                grip_acts = torch.Tensor([[0., 0.]] * self.num_envs).to(self.device)
            else:
                grip_acts = torch.Tensor([[0.04, 0.04]] * self.num_envs).to(self.device)
            # grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * self.num_envs).to(self.device), 
            #                         torch.Tensor([[0.04, 0.04]] * self.num_envs).to(self.device))
            pos_action[:, 7:9] = grip_acts
            franka_dof_pose = self.dof_states[:self.franka_num_dofs,0]
            state = torch.concatenate((self.hand_pos,self.hand_rot,self.root_states[0,0,:7].view(1,-1),franka_dof_pose.view(1,-1)),dim=1).cpu().detach().numpy()
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(effort_action))
            self.run_steps(pre_steps = 5)
            action = torch.concatenate((self.hand_pos,self.hand_rot,pos_action[:,:self.franka_num_dofs]),dim=1).cpu().detach().numpy()

            # save state and video
            if save_video:
                self.gym.render_all_camera_sensors(self.sim)
                # print("Saving video frame:", start_step + step_i)
                step_str = str(start_step + step_i).zfill(4)
                os.makedirs(f"{save_root}/video", exist_ok=True)
                self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.cams[0][0], gymapi.IMAGE_COLOR, f"{save_root}/video/step-{step_str}.png")
                # self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.cams[0][0], gymapi.IMAGE_DEPTH, f"{save_root}/video/depth_step-{step_str}.png")
                # self.gym.write_viewer_image_to_file(self.viewer, f"{save_root}/step-{start_step + step_i}.png")

            points_envs, colors_envs, masks_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, ori_masks_envs = self.get_camera_state()
            point = np.concatenate((points_envs[0][:, :3],colors_envs[0][:, :3]),axis=1)
            ori_point = np.concatenate((ori_points_envs[0][0][:, :3],ori_colors_envs[0][0][:, :3]),axis=1)
            mask = masks_envs[0]
            franka_init_pos = self.franka_reset_pos_list[0]
            franka_init_rot = self.franka_reset_rot_list[0]
            franka_init_pose = np.concatenate((franka_init_pos,franka_init_rot))
            np.savez(os.path.join(save_root,f'{step_str}.npz'),state=state,
                                                               action=action,
                                                               point=point,
                                                               ori_point=ori_point,
                                                               mask=mask,
                                                               depth=depth_envs[0][0],
                                                               init_position=self.init_position,
                                                               franka_init_pose=franka_init_pose)
            
            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(points_envs[0][:, :3])
            # point_cloud.colors = o3d.utility.Vector3dVector(colors_envs[0][:, :3]/255.0)
            # # save_to ply
            # fname = os.path.join(save_root, f"point_cloud_{step_str}.ply")
            # o3d.io.write_point_cloud(fname, point_cloud)

            # franka_dof_pos = self.dof_states[]


            # color_mapping = {
            # 0: [255, 0, 0],   # 红色
            # 1: [255, 0, 0],   # 绿色
            # 2: [255, 0, 0],   # 蓝色
            # 3: [255, 0, 0], # 青色
            # 4: [255, 0, 0], # 紫色
            # 5: [0, 255, 255]  # 黄色
            # }
            # color_mask = np.zeros((seg_envs[0][0].shape[0], seg_envs[0][0].shape[1], 3), dtype=np.uint8)

            # for seg_id, color in color_mapping.items():
            #     color_mask[seg_envs[0][0] == seg_id] = color
            # cv2.imwrite(os.path.join(save_root, f"{step_str}.png"),color_mask)

            # pc = points_envs[0][:, :3]
            # color = colors_envs[0][:, :3]
            # mask = masks_envs[0][:,:]
            # idx = (mask[:,0]==4)
            # robot_pc = pc[idx]
            # robot_color = color[idx]
            # pc_ = o3d.geometry.PointCloud()
            # pc_.points = o3d.utility.Vector3dVector(robot_pc)
            # pc_.colors = o3d.utility.Vector3dVector(robot_color/255.0)
            # o3d.io.write_point_cloud('robot.ply', pc_)

    def move_gripper(self, close_gripper = True, save_video = False, save_root = "", start_step = 0):
        pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        effort_action = torch.zeros_like(pos_action)
        if close_gripper:
            grip_acts = torch.Tensor([[0., 0.]] * self.num_envs).to(self.device)
        else:
            grip_acts = torch.Tensor([[0.04, 0.04]] * self.num_envs).to(self.device)
        # grip_acts = torch.where(close_gripper, torch.Tensor([[0., 0.]] * self.num_envs).to(self.device), 
        #                         torch.Tensor([[0.04, 0.04]] * self.num_envs).to(self.device))
        pos_action[:, :7] = self.robot_dof_qpos_qvel[:,:7,0]
        pos_action[:, 7:9] = grip_acts
        franka_dof_pose = self.dof_states[:self.franka_num_dofs,0]
        state = torch.concatenate((self.hand_pos,self.hand_rot,self.root_states[0,0,:7].view(1,-1),franka_dof_pose.view(1,-1)),dim=1).cpu().detach().numpy()
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(effort_action))
        self.run_steps(pre_steps = 2)
        action = torch.concatenate((self.hand_pos,self.hand_rot,pos_action[:,:self.franka_num_dofs]),dim=1).cpu().detach().numpy()

        
        if save_video:
            self.gym.render_all_camera_sensors(self.sim)
            # print("Saving video frame:", start_step)
            # start_step string, 4 digit
            step_str = str(start_step).zfill(4)
            os.makedirs(f"{save_root}/video", exist_ok=True)
            self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.cams[0][0], gymapi.IMAGE_COLOR, f"{save_root}/video/step-{step_str}.png")

        points_envs, colors_envs, masks_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, ori_masks_envs = self.get_camera_state()
        point = np.concatenate((points_envs[0][:, :3],colors_envs[0][:, :3]),axis=1)
        ori_point = np.concatenate((ori_points_envs[0][0][:, :3],ori_colors_envs[0][0][:, :3]),axis=1)
        mask = masks_envs[0]
        franka_init_pos = self.franka_reset_pos_list[0]
        franka_init_rot = self.franka_reset_rot_list[0]
        franka_init_pose = np.concatenate((franka_init_pos,franka_init_rot))
        franka_init_pose = np.concatenate((franka_init_pos,franka_init_rot))
        np.savez(os.path.join(save_root,f'{step_str}.npz'),state=state,
                                                            action=action,
                                                            point=point,
                                                            ori_point=ori_point,
                                                            mask=mask,
                                                            init_position=self.init_position,
                                                            franka_init_pose=franka_init_pose)
        
        # points_envs, colors_envs, masks_envs, rgb_envs, depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, ori_masks_envs = self.get_camera_state()
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(points_envs[0][:, :3])
        # point_cloud.colors = o3d.utility.Vector3dVector(colors_envs[0][:, :3]/255.0)
        # # save_to ply
        # fname = os.path.join(save_root, f"point_cloud_{step_str}.ply")
        # o3d.io.write_point_cloud(fname, point_cloud)
        
        return start_step + 1
        
    def control_to_pose(self, pose, close_gripper = True, save_video = False, save_root = "", step_num = 0, use_ik = False, start_qpos = None):
        # move to pre-grasp
        self.refresh_observation()
        # print("control:",self.franka_pos)
        USE_IK_CONTROL = use_ik
        if USE_IK_CONTROL:
            self.plan_to_pose_ik(
                torch.tensor(pose[:3], dtype = torch.float32), 
                torch.tensor(pose[3:], dtype = torch.float32),
                close_gripper=close_gripper,
                save_video=save_video,
                save_root = save_root,
                start_step = step_num,
                control_steps = 10
                )
            step_num += 10
            return step_num, None
        else:
            traj = self.plan_to_pose_curobo(
                torch.tensor(pose[:3], dtype = torch.float32), 
                torch.tensor(pose[3:], dtype = torch.float32), 
                start_state=start_qpos
            )
            if traj == None:
                # os.system(f"rm -r {save_root}/video")
                print("traj planning error")
                return step_num, traj
            self.move_to_traj(traj, close_gripper=close_gripper, 
                              save_video=save_video, save_root = save_root, 
                              start_step = step_num
                              )
            step_num += len(traj)
        return step_num, traj

    # not used
    def move_obj_to_pose(self, position, quaternion = None):
        
        root_states = self.root_states.clone()
        root_states[-1, :3] = torch.tensor(position, dtype=torch.float32, device=self.device)
        if quaternion is not None:
            root_states[-1, 3:7] = torch.tensor(quaternion, dtype=torch.float32, device=self.device)
        # self.rb_states[:, self.actor_id, :7] = target_pose
        root_reset_actors_indices = torch.unique(torch.tensor(np.arange(root_states.shape[0]), dtype=torch.float32, device=self.device)).to(dtype=torch.int32)
        res = self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(root_states), gymtorch.unwrap_tensor(root_reset_actors_indices),len(root_reset_actors_indices))
        self.gym.refresh_actor_root_state_tensor(self.sim)
        assert res == True
        self.run_steps(1)


    # not used
    def add_obj_to_env(self, urdf_path, obj_pose_p, final_rotation):
        obj_asset_file = urdf_path
        arti_obj_asset_options = gymapi.AssetOptions()
        arti_obj_asset_options.use_mesh_materials = True
        arti_obj_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        arti_obj_asset_options.override_inertia = True
        arti_obj_asset_options.override_com = True
        arti_obj_asset_options.vhacd_enabled = True
        arti_obj_asset_options.fix_base_link = False
        
        arti_obj_asset_options.vhacd_params = gymapi.VhacdParams()
        arti_obj_asset_options.vhacd_params.resolution = 1000000
        self.num_asset_per_env+=1
        self.obj_assets.append(self.gym.load_asset(self.sim, self.asset_root, obj_asset_file, arti_obj_asset_options))
        self.obj_num_links+=1
        
        for i in range(self.num_envs):
            env = self.envs[i]
            initial_pose = gymapi.Transform()
            initial_pose.p.x = obj_pose_p[0]
            initial_pose.p.y = obj_pose_p[1]
            initial_pose.p.z = obj_pose_p[2]
            rotation_noise = 0.0
            initial_pose.r.x = final_rotation[0]
            initial_pose.r.y = final_rotation[1]
            initial_pose.r.z = final_rotation[2]
            initial_pose.r.w = final_rotation[3]

            self.init_obj_pos_list[i].append([initial_pose.p.x, initial_pose.p.y, initial_pose.p.z])
            self.init_obj_rot_list[i].append([initial_pose.r.x, initial_pose.r.y, initial_pose.r.z, initial_pose.r.w])
            
            added_obj_actor_handle = self.gym.create_actor(env, self.obj_assets[-1], initial_pose, 'added_actor', i, 1, 0)
            
            obj_actor_idx = self.gym.get_actor_rigid_body_index(env, added_obj_actor_handle, 0, gymapi.DOMAIN_SIM)
            self.obj_actor_idxs[i].append(obj_actor_idx)
            self.gym.set_actor_scale(env, added_obj_actor_handle, self.cfgs["obj_scale"])
        self.gym.prepare_sim(self.sim)
        self.init_observation()

    def run_steps(self, pre_steps = 100, refresh_obs = True, print_step = False):
        # simulation loop
        for frame in range(pre_steps):
            if print_step:
                print("Step: ", frame)
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            self.refresh_observation()
            # self.transform_bbox(link_name='link_2')
            
            if not self.headless :
                self.render()
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.step_graphics(self.sim)

        self.refresh_observation()
    
    def clean_up(self):
        # cleanup
        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def _partial_reset(self, to_reset=None):

        """
        reset those need to be reseted
        """

        if to_reset == "all" :
            to_reset = np.ones((self.num_envs,))

        reseted = False
        self.arti_reset_obj_pos_list = []
        self.arti_reset_obj_rot_list = []
        self.franka_reset_pos_list = []
        self.franka_reset_rot_list = []
        # self.arti_reset_obj_rot_list
        for env_id, reset in enumerate(to_reset) :
            # is reset:
            if reset.item() :
                # need randomization
                
                reset_dof_states = self.init_dof_states[env_id].clone()
                reset_root_states = self.root_states[env_id].clone()
                franka_reset_pos_tensor = reset_root_states[0, :3]
                franka_reset_rot_tensor = reset_root_states[0, 3:7]
                franka_reset_dof_pos_tensor = reset_dof_states[:self.franka_num_dofs-2, 0]
                franka_reset_dof_vel_tensor = reset_dof_states[:self.franka_num_dofs-2, 1]
                cabinet_reset_pos_tensor = reset_root_states[2, :3]
                cabinet_reset_rot_tensor = reset_root_states[2, 3:7]
                cabinet_reset_dof_pos_tensor = reset_dof_states[self.franka_num_dofs:, 0]
                cabinet_reset_dof_vel_tensor = reset_dof_states[self.franka_num_dofs:, 1]

                self.intervaledRandom_(franka_reset_pos_tensor, self.franka_reset_position_noise, self.franka_pos_low, self.franka_pos_upper)
                self.intervaledRandom_(franka_reset_rot_tensor, self.franka_reset_rotation_noise)
                self.intervaledRandom_(franka_reset_dof_pos_tensor, self.franka_reset_dof_pos_interval, self.franka_lower_limit_tensor[:7], self.franka_upper_limit_tensor[:7])


                # pos_low = torch.tensor([0.65, 0.20, 0.65]).to(self.device)  # 
                # pos_upper = torch.tensor([0.68, 0.30, 0.75]).to(self.device)
                # pos_low = torch.tensor([0.57, 0.25, 0.60]).to(self.device)  # OpenDoor right
                # pos_upper = torch.tensor([0.63, 0.35, 0.70]).to(self.device)
                
                # pos_low = torch.tensor([0.65, 0.15, 0.60]).to(self.device)  # 
                # pos_upper = torch.tensor([0.68, 0.25, 0.70]).to(self.device)

                # pos_low = torch.tensor([0.58, 0, 0.45]).to(self.device)  
                # pos_upper = torch.tensor([0.60, 0.1, 0.55]).to(self.device)

                # self.cabinet_pos_low = torch.tensor([0.55, 0, 0.45]).to(self.device)  # OpenDrawer
                # self.cabinet_pos_upper = torch.tensor([0.60, 0.1, 0.55]).to(self.device)
                # pos_low = torch.tensor([0.65, -0.06, 0.55]).to(self.device)  
                # pos_upper = torch.tensor([0.75, 0.06, 0.65]).to(self.device)
                self.intervaledRandom_(cabinet_reset_pos_tensor, self.arti_reset_position_noise, self.cabinet_pos_low, self.cabinet_pos_upper)
                self.intervaledRandom_(cabinet_reset_rot_tensor, self.arti_reset_rotation_noise)

                # self.intervaledRandom_(cabinet_reset_pos_tensor, self.arti_reset_position_noise)
                # self.intervaledRandom_(cabinet_reset_rot_tensor, self.arti_reset_rotation_noise)
                self.cabinet_reset_pos_tensor = cabinet_reset_pos_tensor
                print(cabinet_reset_pos_tensor)

                self.arti_reset_obj_pos_list.append(cabinet_reset_pos_tensor.cpu().detach().tolist())
                self.arti_reset_obj_rot_list.append(cabinet_reset_rot_tensor.cpu().detach().tolist())
                self.franka_reset_pos_list.append(franka_reset_pos_tensor.cpu().detach().numpy())
                self.franka_reset_rot_list.append(franka_reset_rot_tensor.cpu().detach().numpy())

                self.root_states[env_id].copy_(reset_root_states)
                self.init_dof_states[env_id].copy_(reset_dof_states)
                reseted = True
        
        if reseted :
            # print("reset:",self.init_dof_states)

            self.gym.set_dof_state_tensor(
                self.sim,
                gymtorch.unwrap_tensor(self.init_dof_states)
            )
            self.gym.set_actor_root_state_tensor(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states)
            )
            self.gym.refresh_dof_state_tensor(self.sim)
            # print("after_reset",self.dof_states)

    def reset(self, to_reset = "all") :

        self._partial_reset(to_reset)

        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        if not self.headless :
            self.render()
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.step_graphics(self.sim)
        
        self.obs_buf = self.refresh_observation()

        return self.obs_buf

    def step(self, actions):

        self._perform_actions(actions)
        
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        if not self.headless :
            self.render()
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.step_graphics(self.sim)

        self.obs_buf = self.refresh_observation()
        success = self._success()
        return self.obs_buf, success


    def render(self, sync_frame_time=False):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)
    
    def _perform_actions(self, actions):

        actions = actions.to(self.device)
        # print(actions)
        
        pos_action = torch.zeros_like(self.dof_pos).squeeze(-1)
        effort_action = torch.zeros_like(pos_action)

        pos_action[:, :9] = actions
        pos_action[:, :9] = tensor_clamp(
                pos_action[:, :9], self.franka_lower_limit_tensor[:9], self.franka_upper_limit_tensor[:9])

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(effort_action))
        self.run_steps(10)

    def _success(self):

        current_part_pos = self.rb_states[self.part_rigid_body_index,:][:3]
        current_part_rot = self.rb_states[self.part_rigid_body_index,:][3:7].cpu().detach().numpy()
        current_part_rot = torch.from_numpy(quaternion2euler(current_part_rot))

        pdist = nn.PairwiseDistance(p=2)
        pos_dist = pdist(current_part_pos, self.init_part_pos)
        rot_dist = pdist(current_part_rot, self.init_part_rot)
        success = (pos_dist > self.pos_dist_bar) | (rot_dist > self.rot_dist_bar)

        # print(success)
        return success

    def intervaledRandom_(self, tensor, dist, lower=None, upper=None) :
        tensor += torch.rand(tensor.shape, device=self.device)*dist*2 - dist
        if lower is not None and upper is not None :
            torch.clamp_(tensor, min=lower, max=upper)


    def cal_handle(self,bbox_id=-1):

        self.bbox_id = bbox_id
        self.manip_name = self.gapart_link_names[0][bbox_id]
        self.part_rigid_body_index = self.gym.find_actor_rigid_body_index(self.envs[0],self.arti_obj_handles[0],self.manip_name,gymapi.DOMAIN_ENV)
        init_rb_states = copy.deepcopy(self.rb_states)
        self.init_part_pos = init_rb_states[self.part_rigid_body_index,:][:3]
        self.init_part_rot = init_rb_states[self.part_rigid_body_index,:][3:7].cpu().detach().numpy()
        self.init_part_rot = torch.from_numpy(quaternion2euler(self.init_part_rot))

        for env_i in range(self.num_envs):
            for gapart_obj_i, gapart_raw_valid_anno in enumerate(self.gapart_raw_valid_annos):
                    
                self.all_bbox_init = self.gapart_init_bboxes[gapart_obj_i]*self.cfgs["asset"]["arti"]["arti_obj_scale"]
                
                rotation = R.from_quat(self.arti_reset_obj_rot_list[env_i])
                rotation_matrix = rotation.as_matrix()
                rotated_bbox_now = np.dot(self.all_bbox_init, rotation_matrix.T)
                
                self.all_bbox_init = rotated_bbox_now + self.arti_reset_obj_pos_list[env_i]
                
                # if not self.headless and True:
                #     idx_set = [[0,1],[1,2],[1,5],[0,4],[0,3],[2,3],[2,6],[3,7],[4,5],[4,7],[5,6],[6,7]]
                #     for part_i in range(len(gapart_raw_valid_anno)):
                #         bbox_now_i = self.all_bbox_init[bbox_id]
                #         for i in range(len(idx_set)):
                #             self.gym.add_lines(self.viewer, self.envs[env_i], 1, 
                #                 np.concatenate((bbox_now_i[idx_set[i][0]], 
                #                                 bbox_now_i[idx_set[i][1]]), dtype=np.float32), 
                #                 np.array([1, 0 ,0], dtype=np.float32))
            
            # manipulate the object with the last part, change it for other objects
            # TODO: change the bbox_id to manipulate parts using annotated semantics
            # get the part bbox and calculate the handle direction
            self.all_bbox_init = torch.tensor(self.all_bbox_init, dtype=torch.float32).to(self.device).reshape(-1, 8, 3)
            self.all_bbox_center_front_face = torch.mean(self.all_bbox_init[:,0:4,:], dim = 1) 
            self.handle_out = self.all_bbox_init[:,0,:] - self.all_bbox_init[:,4,:]
            self.handle_out /= torch.norm(self.handle_out, dim = 1, keepdim=True)
            self.handle_long = self.all_bbox_init[:,0,:] - self.all_bbox_init[:,1,:]
            self.handle_long /= torch.norm(self.handle_long, dim = 1, keepdim=True)
            self.handle_short = self.all_bbox_init[:,0,:] - self.all_bbox_init[:,3,:]
            self.handle_short /= torch.norm(self.handle_short, dim = 1, keepdim=True)
            self.rotations = quaternion_invert(matrix_to_quaternion(torch.cat((self.handle_long.reshape((-1,1,3)), 
                            self.handle_short.reshape((-1,1,3)), -self.handle_out.reshape((-1,1,3))), dim = 1)))
            
            self.init_position = self.all_bbox_center_front_face[bbox_id].cpu().numpy()
            self.handle_out_ = self.handle_out[bbox_id].cpu().numpy()
            self.handle_long_ = self.handle_long[bbox_id].cpu().numpy()
            self.handle_short_ = self.handle_short[bbox_id].cpu().numpy()

            # self.init_position = self.init_position + np.random.uniform(low=-0.04,high=0.04)*self.handle_long_
            
            # self.gym.add_lines(self.viewer, self.envs[env_i], 1, 
            #                         np.concatenate((self.init_position, 
            #                                         self.init_position + 0.35 * self.handle_out_), dtype=np.float32), 
            #                         np.array([0, 0 ,1], dtype=np.float32))

    def motion_planning(self,save_video = True, save_root='record', task_type='PullDrawer'):
            
            rotation = self.rotations[self.bbox_id].cpu().numpy()
            # noisy_rotation = rotation
            std_dev = 0.01
            rot_noise = np.random.normal(0,std_dev,rotation.shape)
            noisy_rotation = rotation + rot_noise
            noisy_rotation = noisy_rotation / np.linalg.norm(noisy_rotation)
            # move the object to the pre-grasp position
            pre_grasp_position = self.init_position + 0.2 * self.handle_out_

            for i in range(1): 
                step_num, traj = self.control_to_pose(
                                    np.array([*pre_grasp_position,*noisy_rotation]), 
                                    close_gripper = False, save_video = save_video, save_root = save_root, step_num = 0, use_ik = False)
                if traj == None:
                    return 0
            # move the object to the grasp position

            for i in range(1): 

                step_num, traj = self.control_to_pose(
                                    np.array([*(self.init_position + (0.182-0.1) * self.handle_out_),*noisy_rotation]), 
                                    close_gripper = False, save_video = save_video, save_root = save_root, step_num = step_num, use_ik = False)
                if traj == None:
                    return 0
                
            # close the grippers
            for i in range(2): 
                self.move_gripper(close_gripper = True, save_video = save_video, save_root = save_root, start_step = step_num)
            
            dict = {"closed_gripper": step_num,
                    "camera_proj": self.cam_projs[0][0].tolist(),
                    "camera_view": self.cam_views[0][0].tolist(),
                    "camera_w": self.cam_w,
                    "camera_h": self.cam_h,
                    "camera_fov": self.horizontal_fov,
                    "cabinet_reset_pos": self.cabinet_reset_pos_tensor.cpu().numpy().tolist(),
                    "franka_base_pos": self.franka_reset_pos_list[0].tolist()}
            
            file = f"{save_root}/closed_gripper.json"
            write_json(dict, file)
            
            # move the object to the lift position
            for i in range(1):  #25

                if task_type == 'OpenDoor':
                    step_num, traj = self.control_to_pose(
                                        np.array([*(self.init_position + (0.1+25*0.01) * self.handle_out_- 0.12*self.handle_short_),*noisy_rotation]), 
                                        close_gripper = True, save_video = save_video, save_root = save_root, step_num = step_num, use_ik = False)
                    if traj == None:
                        return 0
                else:
                    step_num, traj = self.control_to_pose(
                                        np.array([*(self.init_position + (0.1+25*0.01) * self.handle_out_),*noisy_rotation]), 
                                        close_gripper = True, save_video = save_video, save_root = save_root, step_num = step_num, use_ik = False)
                    if traj == None:
                        return 0
                
            return 1
    

    def transform_bbox(self, link_name='link_1'):
        
        link_rigid_body_index = self.gym.find_actor_rigid_body_index(
            self.envs[0],
            self.arti_obj_handles[0],
            link_name,
            gymapi.DOMAIN_ENV
        )

        id = self.gapart_link_names[0].index(link_name)
        bbox_init = self.all_bbox_init[id].cpu().detach().numpy()
        link_pos = self.rb_states[link_rigid_body_index, :][:3].cpu().detach().numpy()
        link_rot = self.rb_states[link_rigid_body_index, :][3:7].cpu().detach().numpy()

        rotation = R.from_quat(link_rot)
        rotation_matrix = rotation.as_matrix()
        rotated_bbox_now = np.dot(bbox_init, rotation_matrix.T)
                
        bbox_now = rotated_bbox_now + link_pos
        
        if not self.headless and True:
            for env_i in range(self.num_envs):
                idx_set = [[0,1],[1,2],[1,5],[0,4],[0,3],[2,3],[2,6],[3,7],[4,5],[4,7],[5,6],[6,7]]
                for i in range(len(idx_set)):
                    self.gym.add_lines(self.viewer, self.envs[env_i], 1, 
                        np.concatenate((bbox_now[idx_set[i][0]], 
                                        bbox_now[idx_set[i][1]]), dtype=np.float32), 
                        np.array([1, 0 ,0], dtype=np.float32))