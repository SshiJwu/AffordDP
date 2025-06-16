from afforddp.utils.vision_model import run_pointsam, run_sam, scale_img_pixel, transfer_pixel
from afforddp.utils.transform import get_image_pixel_from_3d, sample_point_cloud, find_nearest_object_pixel_in_box, get_3d_from_image_pixel, ICP_register,  update_afford
from afforddp.utils.vis import show_results, vis_point_cloud 
from afforddp.featurizer.utils.visualization import IMG_SIZE
from PIL import Image
import numpy as np
import os
import shutil

def affordance_transfer(prompt, gym, memory_buffer, task_name, save_dir, vis_flag=True):
    
    """
    Transfers affordance information from a retrieved source to a target scene using visual and geometric matching.
    
    Parameters:
        prompt (str): Text prompt used for segmentation (e.g., object description for SAM model)
        gym (object): Environment object containing camera and scene information with methods:
                      - cam_projs: Camera projection matrices
                      - cam_views: Camera view matrices
                      - cam_w/cam_h: Camera width/height
                      - get_camera_state(): Returns scene point clouds, RGB, depth, and segmentation
        memory_buffer (object): Memory system that provides retrieval capabilities with methods:
                      - retrieval_id(): Finds matching scene ID based on current RGB and task
                      - get_retrieval_info(): Returns stored RGB, affordance, point cloud, and mask
        task_name (str): Name of the current task for memory retrieval
        save_dir (str): Directory path to save visualization results (images, point clouds)
        vis_flag (bool, optional): Flag to control visualization output. If True (default), 
                      saves visualizations of source/target point clouds with affordances.
    
    Returns:
        None (but saves visualization results to save_dir including:
              - Source/target images with affordance points (always saved)
              - Source/target point clouds with affordance visualization (only if vis_flag=True))
    """
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)
    
    retrieval_id_list = []
    proj = gym.cam_projs[0][0]
    view = gym.cam_views[0][0]
    camera_w = gym.cam_w
    camera_h = gym.cam_h
    points_envs, colors_envs, masks_envs, rgb_envs, \
        depth_envs ,seg_envs, ori_points_envs, ori_colors_envs, ori_masks_envs = gym.get_camera_state()
        
    tgt_point_cloud = points_envs[0]
    tgt_point_color = colors_envs[0]
    tgt_point = np.concatenate((tgt_point_cloud, tgt_point_color), axis=1)
    tgt_mask = masks_envs[0]
    init_scene_img = Image.fromarray(rgb_envs[0][0]).convert('RGB')
    init_depth = depth_envs[0][0].transpose(1,0)
    
    retrieval_id = memory_buffer.retrieval_id(rgb_envs[0][0], task_name)
    retrieval_id_list.append(retrieval_id)
    retrieval_img, retrieval_afford, retrieval_point, retrieval_mask = memory_buffer.get_retrieval_info(retrieval_id)
    contact_point = retrieval_afford[0]
    src_part, src_, src_seg_mask, src_shift, src_scale = run_pointsam(retrieval_point, retrieval_mask, contact_point, 40000)

    if contact_point.ndim != 2:
        contact_point = contact_point.reshape(-1, 3)
    image_pixel = get_image_pixel_from_3d(contact_point, view, proj, camera_w, camera_h)
    retrieval_img.save(f'{save_dir}/src.png')
    show_results(retrieval_img, image_pixel, save_name=f'{save_dir}/src_afford')

    image_pixel = scale_img_pixel(image_pixel, camera_w, camera_h)
    mask_src_array, mask_src_img = run_sam(retrieval_img, prompt)
    mask_tgt_array, mask_tgt_img = run_sam(init_scene_img, prompt)
    tgt_pixel = transfer_pixel(mask_src_img, mask_tgt_img, prompt, image_pixel, ftype='sd_dinov2')
    tgt_pixel = scale_img_pixel(tgt_pixel, IMG_SIZE, IMG_SIZE, new_w=camera_w, new_h=camera_h)

    if mask_tgt_array[tgt_pixel[:,1], tgt_pixel[:,0]] == False:
        tgt_pixel = find_nearest_object_pixel_in_box(mask_tgt_array, tgt_pixel)
    init_scene_img.save(f'{save_dir}/tgt.png')
    show_results(init_scene_img, tgt_pixel, save_name=f'{save_dir}/tgt_afford')
    
    tgt_pixel_depth = init_depth[tgt_pixel[:,0], tgt_pixel[:,1]]
    tgt_pixel_depth = np.array([tgt_pixel_depth]*tgt_pixel.shape[0])
    transfer_point = get_3d_from_image_pixel(tgt_pixel, tgt_pixel_depth, view, proj, camera_w, camera_h)
    tgt_part, tgt_, tgt_seg_mask, tgt_shift, tgt_scale = run_pointsam(tgt_point, tgt_mask, transfer_point, 40000)
    
    T, R, t = ICP_register(src_part, tgt_part, voxel_size=0.025, init="")

    transfer_point = (transfer_point-tgt_shift)/tgt_scale
    retrieval_afford[:, :3] = (retrieval_afford[:, :3]-src_shift)/src_scale

    transfer_afford = update_afford(transfer_point, retrieval_afford, R)
    transfer_afford = transfer_afford*tgt_scale + tgt_shift
    
    if vis_flag :
        obj_pc = tgt_point[tgt_mask[:, 0]==4]

        transfer_afford_color = np.zeros_like(transfer_afford)
        transfer_afford_color[:,0] = 255
        transfer_pc = np.concatenate((transfer_afford, transfer_afford_color), axis=1)

        vis_tgt_pc = np.concatenate((obj_pc, transfer_pc), axis=0)
        vis_point_cloud(vis_tgt_pc, name=f'{save_dir}/tgt_pc')
        
        src_pc = retrieval_point[retrieval_mask[:,0]==4]
        retrieval_afford[:, :3] = retrieval_afford[:, :3]*src_scale + src_shift
        retrieval_afford_color = np.zeros_like(retrieval_afford)
        retrieval_afford_color[:,0] = 255
        retrieval_pc = np.concatenate((retrieval_afford, retrieval_afford_color), axis=1)
        vis_src_pc = np.concatenate((src_pc, retrieval_pc), axis=0)
        vis_point_cloud(vis_src_pc, name=f'{save_dir}/src_pc')
        
    return transfer_afford