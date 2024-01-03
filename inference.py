# -*- coding: utf-8 -*-
'''
@File   :  inference.py
@Time   :  2023/12/21 14:18
@Author :  Yufan Liu
@Desc   :  Process from a PDB file and generate descriptor using trained model, for paired data.
'''

# DO NOT add an inference module in main.py, will make you crazy!

from arguments import parser
from data_prepare import DataPrepare
from trainer import Trainer
from model import MaSIFSearch
import torch
import random
import numpy as np
import os

args = parser.parse_args()
args.data_list = args.data_list.upper()

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

if not args.data_list:
    raise RuntimeError("Must provide a valid PDB file formatted as ABCD_E_F")

if args.processed_path == "data/processed":
    raise UserWarning("you'd better change processed_path to discriminate from training data.")

if not args.cache_model:
    raise RuntimeError("Must specify a trained model ")

prepare = DataPrepare(args, data_list=[args.data_list])  # pass a single PDB each time
prepare.preprocess()

# data part
p1_feat = np.load(os.path.join(args.processed_path, args.data_list, 'p1_input_feat.npy'), allow_pickle=True).item()
p2_feat = np.load(os.path.join(args.processed_path, args.data_list, 'p2_input_feat.npy'), allow_pickle=True).item()

p1_feat = np.concatenate([p1_feat['input_feature'], 
                          np.expand_dims(p1_feat['rho'], 2), 
                          np.expand_dims(p1_feat['theta'], 2)], axis=2)
p1_feat[np.isnan(p1_feat)] = 0

p2_feat = np.concatenate([p2_feat['input_feature'], 
                          np.expand_dims(p2_feat['rho'], 2), 
                          np.expand_dims(p2_feat['theta'], 2)], axis=2)
p2_feat[np.isnan(p2_feat)] = 0


p1_forward = torch.from_numpy(p1_feat).to(args.device).to(torch.float32)
p2_forward = torch.from_numpy(p2_feat).to(args.device).to(torch.float32)
p1_reverse = flip_feature(p1_feat).to(torch.float32)
p2_reverse = flip_feature(p2_feat).to(torch.float32)

# essential part for training
model = MaSIFSearch(args).to(args.device)
model.load_state_dict(torch.load(args.cache_model))

descriptors = []
model.eval()
with torch.no_grad():
    for feat in [p1_forward, p2_forward, p1_reverse, p2_reverse]:
        desc = model((feat, feat, feat))[0].detach().cpu().numpy()
        descriptors.append(desc)

# save to folder
np.save(os.path.join(args.processed_path, args.data_list, 'p1_forward.npy'), descriptors[0])
np.save(os.path.join(args.processed_path, args.data_list, 'p2_forward.npy'), descriptors[1])
np.save(os.path.join(args.processed_path, args.data_list, 'p1_reverse.npy'), descriptors[2])
np.save(os.path.join(args.processed_path, args.data_list, 'p2_reverse.npy'), descriptors[3])


