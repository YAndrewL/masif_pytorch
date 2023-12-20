# -*- coding: utf-8 -*-
'''
@File   :  inference.py
@Time   :  2023/12/15 14:09
@Author :  Yufan Liu
@Desc   :  Generate features for protein surface of a given PDB file, a script.
'''


from arguments import parser
from data_prepare import DataPrepare
from trainer import Trainer
from model import MaSIFSearch
import torch
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

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

    pos_dist = np.save("./experiments/pos_dist.npy", pos)
    neg_dist = np.save("./experiments/neg_dist.npy", neg)


    labels = np.concatenate([np.ones((len(pos))), np.zeros((len(neg)))])
    dist_pairs = np.concatenate([pos, neg])
    return roc_auc_score(labels, dist_pairs)   


training_list = open("data/list/train_all.txt").readlines()
training_list = [x.strip() for x in training_list]

testing_list = open("data/list/test_all.txt").readlines()
testing_list = [x.strip() for x in testing_list]

prepare = DataPrepare(args, 
                      training_list=training_list, 
                      testing_list=testing_list)

# dataset
train_set = prepare.dataset(data_type='train',
                            batch_size=args.batch_size,
                            pair_shuffle=args.pair_shuffle)
val_set = prepare.dataset(data_type='val',
                            batch_size=args.batch_size,
                            pair_shuffle=args.pair_shuffle)
test_set = prepare.dataset(data_type='test',
                            batch_size=args.batch_size,
                            pair_shuffle=args.pair_shuffle)

# essential part for training
model = MaSIFSearch(args).to('cuda')
model.load_state_dict(torch.load("model/benchmark/12-18-20-20/model.pth"))

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

