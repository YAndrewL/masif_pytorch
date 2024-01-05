# -*- coding: utf-8 -*-
'''
@File   :  screen.py
@Time   :  2024/01/03 19:14
@Author :  Yufan Liu
@Desc   :  Screen target pocket with a prepared library
'''

# target KRAS, with peptide library


# test
from arguments import parser
from data_prepare import process_single
import sys
import torch
import random
import numpy as np
import os
from model import MaSIFSearch
from tqdm import tqdm

args = parser.parse_args()

# following part is process single, use separately.
# pdb = args.data_list
# process_single(args, pdb, 'analysis/KRAS/raw_files', 'analysis/KRAS/raw_feature')

# target: 7YV1_A, library: peptide

torch.manual_seed(args.random_seed)
random.seed(args.random_seed)
np.random.seed(args.random_seed)
pi = torch.tensor(3.141592653589793)


def flip_feature(binder):
    binder = torch.from_numpy(binder).to(args.device)
    b_feat = []
    for i in range(7):
        if i == 3 or i == 5:   
            b_feat.append(binder[:, :, i].unsqueeze(-1))
        elif i == 6:
            # theta
            feat = 2 * pi - binder[:, :, i]
            b_feat.append(feat.unsqueeze(-1))
        else:
            feat = -binder[:, :, i]
            b_feat.append(feat.unsqueeze(-1))
    binder = torch.cat(b_feat, dim=-1)
    return binder

model = MaSIFSearch(args).to(args.device)
model.load_state_dict(torch.load(args.cache_model))
#for name in tqdm(os.listdir("peptide_library/processed_features")):
#args.data_list = name

feat = np.load("analysis/KRAS/raw_feature/7YV1_A_input_feat.npy", allow_pickle=True).item()
feat = np.concatenate([feat['input_feature'], 
                        np.expand_dims(feat['rho'], 2), 
                        np.expand_dims(feat['theta'], 2)], axis=2)
feat[np.isnan(feat)] = 0
feat = flip_feature(feat).to(torch.float32)
#feat = torch.from_numpy(feat).to(args.device).to(torch.float32) 

# essential part for training


model.eval()
with torch.no_grad():
    desc = model((feat, feat, feat))[0].detach().cpu().numpy()

np.save('analysis/KRAS/raw_feature/7YV1_A_desc.npy', desc)


