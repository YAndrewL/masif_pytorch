# -*- coding: utf-8 -*-
'''
@File   :  run_test.py
@Time   :  2023/12/22 23:01
@Author :  Yufan Liu
@Desc   :  Run test set
'''

import os
from arguments import parser
from data_prepare import DataPrepare
from trainer import Trainer
from model import MaSIFSearch
import torch
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from preprocessing.cacheSample import generate_data_cache

args = parser.parse_args()

torch.manual_seed(args.random_seed)
random.seed(args.random_seed)
np.random.seed(args.random_seed)

# prepare data
# note: execute prepare.preprocess in advance due to speed issue


def compute_loss(outputs):
    # descriptos
    binder, pos, neg = outputs
    dist_p = dist(binder, pos)
    dist_n = dist(neg, binder)

    score = (dist_p, dist_n)

    return score

def dist(a, b):
    assert a.shape == b.shape
    return torch.sum(torch.square(a - b), 1)


def compute_roc_auc(pos, neg):
    pos = pos.detach().cpu().numpy()
    neg = neg.detach().cpu().numpy()

    pos_dist = np.save("./results/masif_pos_dist.npy", pos)
    neg_dist = np.save("./results/masif_neg_dist.npy", neg)


    labels = np.concatenate([np.ones((len(pos))), np.zeros((len(neg)))])
    dist_pairs = np.concatenate([pos, neg])
    return roc_auc_score(labels, dist_pairs)   

# cache as training
# all list
test_list = [x.strip() for x in os.listdir("./peptide_data/processed")]
#generate_data_cache(args, './peptide_dataset/test', test_list)

prepare = DataPrepare(args)

test_set = prepare.dataset(data_type='test',
                            batch_size=args.batch_size,
                            pair_shuffle=args.pair_shuffle)

# essential part for training
model = MaSIFSearch(args).to('cuda')
model.load_state_dict(torch.load("experiments/masif_logp/12-21-16-35/model.pth"))

model.eval()
with torch.no_grad():
    loss = []
    score = []
    for data in tqdm(test_set):
        outputs = model(data)
        score_ = compute_loss(outputs)
        score.append(score_)
    # score: [(pos:, neg), ...]
    pos = torch.cat([d[0] for d in score])  # [N-sample,]
    neg = torch.cat([d[1] for d in score])

    roc = 1 - compute_roc_auc(pos, neg)
    print(f"AUC-ROC is {roc}")

