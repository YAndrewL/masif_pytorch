# -*- coding: utf-8 -*-
'''
@File   :  dataset.py
@Time   :  2023/12/14 21:08
@Author :  Yufan Liu
@Desc   :  Dataset class
'''

from torch.utils.data import Dataset
import torch
import numpy as np
import os
import random
pi = torch.tensor(3.141592653589793)


class SurfaceDataset(Dataset):
    def __init__(self, args, dataset_type:str, pair_shuffle: bool):
        super().__init__()
        assert dataset_type in ['train', 'val', 'test']
        self.dataset_type = dataset_type
        self.args = args
        self.path = os.path.join(args.dataset_path, dataset_type)
        
        self.binder = np.load(os.path.join(self.path, 'binder_feature.npy'))
        self.positive = np.load(os.path.join(self.path, 'positive_feature.npy'))
        self.negative = np.load(os.path.join(self.path, 'negative_feature.npy'))

        self.data_len = self.binder.shape[0]
        self.pair_shuffle = pair_shuffle

    def __getitem__(self, index):
        # shuffle list every time for positive and negative
        if self.pair_shuffle:
            pidx = random.randint(0, self.data_len - 1)
            nidx = random.randint(0, self.data_len - 1)
        else:
            pidx = index
            nidx = index

        binder = torch.from_numpy(self.binder[index]).unsqueeze(0).to(torch.float32)
        pos = torch.from_numpy(self.positive[pidx]).unsqueeze(0).to(torch.float32)
        neg = torch.from_numpy(self.negative[nidx]).unsqueeze(0).to(torch.float32)

        return binder, pos, neg
    
    def __len__(self):
        return self.data_len
    
class InferenceSet(Dataset):
    """a simple implementation for inference
    """
    def __init__(self, file_path):
        # file path should be in npy directly. 
        super().__init__()
        
        self.data = np.load(file_path)
        # should be ndarray with shape [N, vertex, 7]
        self.data_len = self.data.shape[0]

    def __getitem__(self, index):
        data = torch.from_numpy(self.data[index]).unsqueeze(0)
        # for consistent with surfacedata, easy to use collate function
        return data, data, data  
    
    def __len__(self):
        return self.data_len


def collate_fn(flip=True):
    # wrapper
    def collate(batch):
        # collate tuple
        binder = torch.cat([x[0] for x in batch], dim=0)
        pos = torch.cat([x[1] for x in batch], dim=0)
        neg = torch.cat([x[2] for x in batch], dim=0)

        # batch size =1
        if len(binder.shape) == 2:
            binder = binder.unsqueeze(0)
        if len(pos.shape) == 2:
            pos = pos.unsqueeze(0)        
        if len(neg.shape) == 2:
            neg = neg.unsqueeze(0)

        # flip binder, except logp
        # feature order: shapeindex, ddc, charge, logp, apbs, rho, theta
        # for rho and theta, rho is fixed, but theta need to relfect in 2 pi
        if flip:
            b_feat = []
            for i in range(7):
                if i == 3 or i == 5:
                    b_feat.append(binder[:, :, i].unsqueeze(-1))
                elif i == 6:
                    # rho
                    feat = 2 * pi - binder[:, :, i]
                    b_feat.append(feat.unsqueeze(-1))
                else:
                    feat = -binder[:, :, i]
                    b_feat.append(feat.unsqueeze(-1))
        binder = torch.cat(b_feat, dim=-1)
        return binder, pos, neg
    return collate