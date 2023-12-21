# -*- coding: utf-8 -*-
'''
@File   :  inference.py
@Time   :  2023/12/21 14:18
@Author :  Yufan Liu
@Desc   :  Process from a PDB file and generate descriptor using trained model
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
p1_feat = np.load(os.path.join(args.processed_path, args.data_list, 'p1_input_feat.npy'))
p2_feat = np.load(os.path.join(args.processed_path, args.data_list, 'p2_input_feat.npy'))

p1_forward = torch.from_numpy(p1_feat).to(args.device)
p1_forward = torch.from_numpy(p2_feat).to(args.device)
p1_reverse = flip_feature(p1_feat)
p2_reverse = flip_feature(p2_feat)

# essential part for training
model = MaSIFSearch(args).to(args.device)
