# -*- coding: utf-8 -*-
'''
@File   :  main.py
@Time   :  2023/12/07 14:52
@Author :  Yufan Liu
@Desc   :  Main function for training/inference
'''
from arguments import parser
from data_prepare import DataPrepare
from trainer import Trainer
from model import MaSIFSearch
import torch
import random
import numpy as np


args = parser.parse_args()

torch.manual_seed(args.random_seed)
random.seed(args.random_seed)
np.random.seed(args.random_seed)

# prepare data
# note: execute prepare.preprocess in advance due to speed issue

if args.prepare_data:
    prepare = DataPrepare(args, data_list=[args.data_list])  # pass a single PDB each time
    prepare.preprocess()
    prepare.cache()
    exit(0)

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
model = MaSIFSearch(args)

trainer = Trainer(args=args, 
                  model=model,
                  datasets=[train_set, val_set, test_set])
trainer.train()
