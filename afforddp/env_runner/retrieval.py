import os
import glob
import torch
import json
import numpy as np
import h5py
from PIL import Image
import torch.nn as nn
from diffusion_policy_3d.vision.GroundedSAM.grounded_sam_utils import prepare_gsam_model, inference_one_image, crop_image
from transformers import CLIPModel, CLIPProcessor

def contains_zarr(s):
    return 'zarr' in s

class MemoryBank:

    def __init__(self, data_dir="/home/user/Downloads/NAS/sim_data"):

        memory_file = f"{data_dir}/Memory/memory.json"

        self.init_clip()
        self.init_gsam()
        dirs = os.listdir(data_dir)

        if os.path.exists(memory_file):
            
            with open(memory_file, 'r', encoding='utf-8') as file:
                self.memory = json.load(file)

        else:
            memory_dict = {}
            for task_dir in dirs:

                if contains_zarr(task_dir) or (task_dir=='Memory'):
                        continue
                task_name = task_dir
                task_dict = {}
                instance_dirs = os.listdir(f"{data_dir}/{task_dir}")

                for instance_dir in instance_dirs:

                    if contains_zarr(instance_dir) or (instance_dir=='record_data'):
                        continue
                    instance_dir_abs = os.path.join(data_dir, task_dir, instance_dir)
                    tra_datas = os.listdir(instance_dir_abs)

                    for tra_data in tra_datas:

                        if tra_data == 'config.json':
                            continue

                        id = f"{instance_dir}_{tra_data}"
                        # tra_data = np.load(os.path.join(tra_path, "0000.npz"), "r")
                        if task_name == 'OpenDoor_part_right':
                            image = Image.open(f"{instance_dir_abs}/{tra_data}/video/step-0050.png").convert('RGB')
                        else:
                            image = Image.open(f"{instance_dir_abs}/{tra_data}/video/step-0000.png").convert('RGB')
                        image = Image.open(f"{instance_dir_abs}/{tra_data}/video/step-0000.png").convert('RGB')
                        cropped_img = self.run_gsam(image)
                        cropped_img_feat = self.img_extractor(cropped_img).tolist()
                        cropped_img_PIL = Image.fromarray(cropped_img).save(f'vis_crop_img/{id}.png')
                        task_dict[id] = cropped_img_feat

                memory_dict[task_name] = task_dict

            os.makedirs(f"{data_dir}/Memory")

            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_dict, f, ensure_ascii=False, indent=4)

            self.memory = memory_dict

    def init_clip(self):

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def img_extractor(self, img):

        inputs = self.clip_processor(images=img, return_tensors="pt", padding=True)
        img_feat = self.clip_model.get_image_features(**inputs)
        return img_feat
    
    def init_gsam(self):

        self.grounded_dino_model, self.sam_predictor = prepare_gsam_model(device="cuda")

    def run_gsam(self, img, prompt='cabinet', box_threshold = 0.3, text_threshold=0.25):

        # self.init_gsam()
        if not isinstance(img, Image.Image):
            PIL_img = Image.fromarray(img).convert('RGB')
        else:
            PIL_img = img

        tgt_masks = inference_one_image(np.array(PIL_img), 
                                    self.grounded_dino_model, 
                                    self.sam_predictor, 
                                    box_threshold=box_threshold, 
                                    text_threshold=text_threshold, 
                                    text_prompt=prompt, device="cuda").cpu().numpy()
        mask = np.repeat(tgt_masks[0,0][:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        img_masked = np.array(PIL_img) * mask + 255 * (1 - mask)
        mask_PIL_img = np.array(Image.fromarray(img_masked).convert('RGB'))
        cropped_img, cropped_mask, _ = crop_image(mask_PIL_img, mask, margin=0)

        return cropped_img

    def retrieval_id(self, target_img, task_name):

        self.init_clip()
        self.init_gsam()
        
        best = 0
        cos = nn.CosineSimilarity(dim=0)
        cropped_tgt_img = self.run_gsam(target_img)
        tgt_img_feat = self.img_extractor(cropped_tgt_img)
        
        task_memory = self.memory[task_name]
        retrieval_ids = task_memory.keys()
        for retrieval_id in retrieval_ids:
            clip_feat = torch.tensor(task_memory[retrieval_id])
            cos = nn.CosineSimilarity(dim=1)
            sim = cos(tgt_img_feat, clip_feat).item()
            if sim>best:
                best_id = retrieval_id
                best = sim

        del self.grounded_dino_model
        del self.sam_predictor
        del self.clip_model
        del self.clip_processor

        return best_id


        # print('best_similarity:',sim)
        # print('best_id:',best_id)

    def get_retrieval_info(self, id, retrieval_path, img_id='0000'):

        obj_id = id.split("_")[0]
        traj_id = id.split("_")[-1]

        img_path = glob.glob(f"{retrieval_path}/{obj_id}/*_{traj_id}")[0]
        img = Image.open(f"{img_path}/video/step-{img_id}.png").convert('RGB')
        data_path = glob.glob(f"{retrieval_path}/record_data/{obj_id}/*_{traj_id}.npz")[0]
        retrieval_data = np.load(f"{data_path}")
        afford = retrieval_data['afford'][0]
        point = retrieval_data['point'][0]
        mask = retrieval_data['mask'][0]

        return img, afford, point, mask



if __name__ == '__main__':

    buffer = MemoryBank(data_dir="/home/user/Downloads/NAS/sim_data_2")
    target_img = np.array(Image.open("/home/user/Downloads/NAS/record/OpenDoor/41083_traj_58/video/step-0000.png").convert("RGB"))
    # target_img = np.array(Image.open("/home/user/Downloads/NAS/record/OpenDrawer/45661_45661_traj_62/video/step-0000.png").convert("RGB"))
    buffer.retrieval(target_img, 'OpenDrawer')
