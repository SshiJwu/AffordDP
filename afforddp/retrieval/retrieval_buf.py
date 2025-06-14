import os
import sys
sys.path.append(os.getcwd())
from third_party.GroundedSAM.grounded_sam_utils import prepare_gsam_model, inference_one_image, crop_image
from transformers import CLIPModel, CLIPProcessor

import os
import glob
from PIL import Image
import numpy as np
import json
import shutil
import torch
import torch.nn as nn

class RetrievalBuf:
    
    def __init__(self, data_dir=None, save_dir=None, task_name=None):
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.retrieval_path = f"{self.data_dir}/{task_name}"
        memory_file = f"{save_dir}/memory.json"
            
        self.init_clip()
        self.init_gsam()
        
        if os.path.exists(memory_file):
            with open(memory_file, 'r', encoding='utf-8') as file:
                self.memory = json.load(file)
        else:
            self.memory = self.extract_memory_data()
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=4)
            

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

    def extract_memory_data(self):
        
        memory_dict = {}
        
        # if not os.path.exists("vis_crop_img"):
        #     os.makedirs("vis_crop_img")
        # else:
        #     shutil.rmtree("vis_crop_img")
        #     os.makedirs("vis_crop_img")
            
        for task_dir in os.listdir(self.data_dir):
            
            task_name = task_dir
            task_dict = {}
            
            instance_dirs = os.listdir(f"{self.data_dir}/{task_dir}")
            instance_dirs = [d for d in instance_dirs if os.path.isdir(os.path.join(f"{self.data_dir}/{task_dir}", d))]

            for instance in instance_dirs:
                traj_dirs =  os.listdir(f"{self.data_dir}/{task_dir}/{instance}")
                traj_dirs = [d for d in traj_dirs if os.path.isdir(os.path.join(f"{self.data_dir}/{task_dir}/{instance}", d))]
                for traj in traj_dirs:
                    id = f"{instance}_{traj}"
                    image = Image.open(f"{self.data_dir}/{task_dir}/{instance}/{traj}/video/step-0000.png").convert('RGB')
                    cropped_img = self.run_gsam(image)
                    cropped_img_feat = self.img_extractor(cropped_img).tolist()
                    cropped_img_PIL = Image.fromarray(cropped_img).save(f'vis_crop_img/{id}.png')
                    task_dict[id] = cropped_img_feat
            
            memory_dict[task_name] = task_dict
        
        return memory_dict
    
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
            if sim > best:
                best_id = retrieval_id
                best = sim

        del self.grounded_dino_model
        del self.sam_predictor
        del self.clip_model
        del self.clip_processor

        return best_id
        
    def get_retrieval_info(self, id, retrieval_path=None, img_id='0000'):

        obj_id = id.split("_")[0]
        traj_id = id.split("_")[-1]
        if retrieval_path is None:
            retrieval_path = self.retrieval_path

        img_path = glob.glob(f"{retrieval_path}/{obj_id}/*_{traj_id}")[0]
        img = Image.open(f"{img_path}/video/step-{img_id}.png").convert('RGB')
        data_path = glob.glob(f"{retrieval_path}/{obj_id}/*_{traj_id}.npz")[0]
        retrieval_data = np.load(f"{data_path}")
        afford = retrieval_data['afford'][0]
        point = retrieval_data['raw_point'][0]
        mask = retrieval_data['raw_mask'][0]

        return img, afford, point, mask
        
if "__main__" == __name__:
    
    memory_buf = RetrievalBuf(data_dir="record",
                              save_dir="data/memory")