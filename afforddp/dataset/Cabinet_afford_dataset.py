import os
import sys
sys.path.append(os.getcwd())
from typing import Dict
import torch
import numpy as np
import copy
from afforddp.common.pytorch_util import dict_apply
from afforddp.dataset.base_dataset import BasePointcloudDataset
from afforddp.common.replay_buffer import ReplayBuffer
from afforddp.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from afforddp.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

class CabinetManipAffordDataset(BasePointcloudDataset):
    def __init__(
            self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.00,
            max_train_episodes=None
    ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point', 'mask', 'img', 'init_pos', 'pose', 'afford'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.data_dir = os.path.dirname(zarr_path)
        # print()
        
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'][...,7:],
            'state': self.replay_buffer['state'][...,:],
            'afford': self.replay_buffer['afford']
            # 'point_cloud': self.replay_buffer['point_cloud'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()

        return normalizer

    def __len__(self) -> int:
        # print("~~~~~~~~~~~~~~~~~~~~")
        # print(len(self.sampler))
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # 
        point_cloud = sample['point'][:,:].astype(np.float32) # 
        
        data = {
            'obs': {
                'point_cloud': point_cloud, # 
                'state': agent_pos, #
            },
            'afford': sample['afford'].astype(np.float32),
            'action': sample['action'][:,7:].astype(np.float32) # T, D_action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
    

