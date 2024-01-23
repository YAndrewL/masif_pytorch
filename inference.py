# -*- coding: utf-8 -*-
'''
@File   :  inference.py
@Time   :  2023/12/21 14:18
@Author :  Yufan Liu
@Desc   :  Process from a PDB file and generate descriptor using trained model, for paired data.
'''

import os
from arguments import parser
from model import MaSIFSearch
import torch
import random
import numpy as np
import time
from tqdm import tqdm

args = parser.parse_args()

torch.manual_seed(args.random_seed)
random.seed(args.random_seed)
np.random.seed(args.random_seed)
pi = torch.tensor(3.141592653589793)


def flip_feature(binder):
    binder = torch.from_numpy(binder).to(args.device).to(torch.float32)
    b_feat = []
    for i in range(7):
        if i == 3 or i == 5:   
            b_feat.append(binder[:, :, i].unsqueeze(-1))
        elif i == 6:
            # theta
            feat = 2 * pi - binder[:, :, i]
            b_feat.append(feat.unsqueeze(-1))
        elif i == 2:
            # lyf hack for chemical net, feature 2 is now atom type.
            feat = binder[:, :, i]
            b_feat.append(feat.unsqueeze(-1))
        else:
            feat = -binder[:, :, i]
            b_feat.append(feat.unsqueeze(-1))
    binder = torch.cat(b_feat, dim=-1)
    return binder

tic = time.time()
# use a total file to do library in batch
model = MaSIFSearch(args).to(args.device)
model.load_state_dict(torch.load(args.cache_model))

target = np.load(args.inf_target_path)
# target need to be flip in both dataset and chemical net
target = flip_feature(target)

binder = np.load(args.inf_binder_path)
#binder = torch.from_numpy(binder).to(args.device).to(torch.float32)
binder = flip_feature(binder)
print(f"feature loaded. Take time {time.time() - tic}")

# too large, chunk the data
#n_chunks = binder.size(0) // args.batch_size + (1 if binder.size(0) & args.batch_size else 0)
#binders = torch.chunk(binder, n_chunks)
#print("length of chunkcs:", len(binders))

# forward
binder_desc = []
model.eval()
with torch.no_grad():
    target_desc = model((target, target, target))[0].cpu().numpy()  # flipped feature
    # for b in tqdm(binders):
    #     binder_desc.append(
    #         model((b, b, b))[1].cpu().numpy()
    #     )
    # binder_desc = np.concatenate(binder_desc, 0)
    binder_desc = model((binder, binder, binder))[0].cpu().numpy()

# save
np.save(os.path.join(args.inf_save_path, 'mras_desc.npy'), target_desc)
np.save(os.path.join(args.inf_save_path, 'shoc2_desc.npy'), binder_desc)
print("completed.")

