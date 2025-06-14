import torch
import numpy as np
import open3d as o3d
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import PILToTensor
from PIL import Image, ImageDraw
from third_party.GroundedSAM.grounded_sam_utils import prepare_gsam_model, inference_one_image
from point_sam import build_point_sam
from afforddp.featurizer.utils.visualization import IMG_SIZE, Demo
from afforddp.featurizer import SDFeaturizer, DINOFeaturizer, CLIPFeaturizer, DINOv2Featurizer, RADIOFeaturizer, SD_DINOv2Featurizer
from afforddp.utils.transform import sample_point_cloud

featurizers = {
    'sd': SDFeaturizer,
    'clip': CLIPFeaturizer,
    'dino': DINOFeaturizer,
    'dinov2': DINOv2Featurizer,
    'radio': RADIOFeaturizer,
    'sd_dinov2': SD_DINOv2Featurizer
}

def run_pointsam(points, masks, contact_point, num_sample_point):

    if contact_point.ndim != 1:
        contact_point = contact_point.reshape(-1)

    ckpt_path = "third_party/Point_SAM/pretrained/model.safetensors"
    group_number = 2048
    group_size = 512

    model = build_point_sam(ckpt_path, group_number, group_size).cuda()
        
    points = points[masks[:, 0]==4] # object pcd
    points = sample_point_cloud(points, num_sample_point)
    xyz = points[:, :3]
    rgb = points[:, 3:6] / 255
    
    # normalize
    p_shift = xyz.mean(0)
    p_scale = np.linalg.norm(xyz - p_shift, axis=-1).max()
    xyz = (xyz - p_shift) / p_scale
    contact_point = (contact_point - p_shift) / p_scale
    
    # set pcsam variables
    pc_xyz, pc_rgb = (
        torch.from_numpy(xyz).cuda().float(),
        torch.from_numpy(rgb).cuda().float(),
    )
    prompt = torch.tensor(np.array([[contact_point]])).cuda().float()
    pc_xyz, pc_rgb = pc_xyz.unsqueeze(0), pc_rgb.unsqueeze(0)
    model.set_pointcloud(pc_xyz, pc_rgb)
    mask, iou_preds, logits = model.predict_masks(prompt, torch.tensor([[1]]).cuda())
    model.clear()
    
    segment_mask = mask[0][torch.argmax(iou_preds[0])] > 0
    
    # draw(pc_xyz.squeeze(), "npz"+name+"_contact_point", None, contact_point)
    # draw(pc_xyz.squeeze(), "npz"+name, np.array(segment_mask.cpu().numpy().tolist()), contact_point)
    # print(name,"done.")
    
    part = pc_xyz.squeeze()[segment_mask]
    part = part.cpu().numpy()
    prompt = prompt.cpu().numpy()
    pc_xyz = pc_xyz.squeeze().squeeze().cpu().numpy()
    segment_mask = segment_mask.squeeze().cpu().numpy()
    # normalize the part
    shift_prime = part.mean(0)
    scale_prime = np.linalg.norm(part - shift_prime, axis=-1).max()
    part = (part - shift_prime) / scale_prime
    pc_xyz = (pc_xyz - shift_prime) / scale_prime
    prompt = (prompt - shift_prime) / scale_prime
    return part, pc_xyz, segment_mask, p_shift+shift_prime*p_scale, p_scale*scale_prime

def extract_ft(img: Image.Image, prompt=None, ftype='sd_dinov2'):
    '''
    preprocess of img to `img`:
    img = Image.open(filename).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    '''
    if img.size != (IMG_SIZE, IMG_SIZE):
        img = img.resize((IMG_SIZE, IMG_SIZE))
    img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2 # C, H, W
    img_tensor = img_tensor.unsqueeze(0).cuda() # 1, C, H, W

    assert ftype in ['sd', 'clip', 'dino', 'dinov2', 'radio', 'sd_dinov2']
    featurizer = featurizers[ftype]()
    
    ft = featurizer.forward(
        img_tensor,
        block_index=1, # only for clip & dino
        prompt=prompt, # only for sd
        ensemble_size=2 # only for sd
    )
    del featurizer

    return ft

def match_fts(src_ft, tgt_ft, pos, src_img_PIL, tgt_img_PIL, save_root=None):

    num_channel = src_ft.size(1)
    src_ft = nn.Upsample(size=(IMG_SIZE, IMG_SIZE), mode='bilinear')(src_ft)
    tgt_ft = nn.Upsample(size=(IMG_SIZE, IMG_SIZE), mode='bilinear')(tgt_ft)
    x, y = pos[0], pos[1]
    x_norm = 2 * x / (IMG_SIZE - 1) - 1
    y_norm = 2 * y / (IMG_SIZE - 1) - 1
    src_vec = torch.nn.functional.grid_sample(src_ft, torch.tensor([[[[x_norm, y_norm]]]]).float().cuda(), align_corners=True).squeeze(2).squeeze(2)
    tgt_vecs = tgt_ft.view(1, num_channel, -1) # 1, C, H*W
    src_vec = F.normalize(src_vec) # 1, C
    tgt_vecs = F.normalize(tgt_vecs) # 1, C, HW
    cos_map = torch.matmul(src_vec, tgt_vecs).view(1, IMG_SIZE, IMG_SIZE).cpu().numpy() # 1, H, W
    max_xy, _ = sample_highest(cos_map)
    # max_xy = (max_xy[0] * tgt_img_PIL.size[0] / IMG_SIZE, max_xy[1] * tgt_img_PIL.size[1] / IMG_SIZE)
    del tgt_vecs
    del src_vec
    return max_xy

def sample_highest(cos_map: np.ndarray):
    max_yx = np.unravel_index(cos_map[0].argmax(), cos_map[0].shape)
    max_xy = np.array([max_yx[1], max_yx[0]]).reshape(1,-1)
    return max_xy, cos_map[0][max_yx]

def run_sam(PIL_img, prompt):

    box_threshold = 0.3
    text_threshold = 0.25
    grounded_dino_model, sam_predictor = prepare_gsam_model(device="cuda")
    tgt_masks = inference_one_image(np.array(PIL_img), 
                                    grounded_dino_model, 
                                    sam_predictor, 
                                    box_threshold=box_threshold, 
                                    text_threshold=text_threshold, 
                                    text_prompt=prompt, device="cuda").cpu().numpy()
    mask = np.repeat(tgt_masks[0,0][:, :, np.newaxis], 3, axis=2).astype(np.uint8)
    img_masked = np.array(PIL_img) * mask + 255 * (1 - mask)
    mask_PIL_img = Image.fromarray(img_masked).convert('RGB')


    del grounded_dino_model
    del sam_predictor

    return tgt_masks[0,0], mask_PIL_img

def transfer_pixel(src_img_PIL, tgt_img_PIL, prompt, points, save_root=None, ftype='sd_dinov2'):

    transfer_pixels = []
    src_ft = extract_ft(src_img_PIL, prompt=prompt, ftype=ftype)
    tgt_ft = extract_ft(tgt_img_PIL, prompt=prompt, ftype=ftype)
    for point in  points:
        transfer_pixel = match_fts(src_ft, tgt_ft, pos=point, src_img_PIL=src_img_PIL, tgt_img_PIL=tgt_img_PIL)
        transfer_pixels.append(transfer_pixel)

    transfer_pixels = np.concatenate(transfer_pixels, axis=0)

    return transfer_pixels

def scale_img_pixel(img_pixel, original_w, original_h, new_w=IMG_SIZE, new_h=IMG_SIZE):

    scale_width = new_w / original_w
    scale_height = new_h / original_h

    img_pixel[:,0] = img_pixel[:,0] * scale_width
    img_pixel[:,1] = img_pixel[:,1] * scale_height

    return img_pixel

def run_demo(src_path, tgt_path, prompt, ftype):
    file_list = [src_path, tgt_path]
    img_list = []
    ft_list = []
    for filename in file_list:
        img = Image.open(filename).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_list.append(img)
        ft = extract_ft(img, prompt, ftype)
        ft_list.append(ft)
    
    ft = torch.cat(ft_list, dim=0)
    demo = Demo(img_list, ft, IMG_SIZE)
    demo.plot_img_pairs(fig_size=5)