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
import os

args = parser.parse_args()

torch.manual_seed(args.random_seed)
if args.device != 'cpu':
    torch.cuda.manual_seed(args.random_seed)
random.seed(args.random_seed)
np.random.seed(args.random_seed)

# prepare data
# note: execute prepare.preprocess in advance (for the first time) due to speed issue

if args.prepare_data:
    print("Data prepare mode, no model will be trained.")
    prepare = DataPrepare(args, data_list=[args.data_list])  # pass a single PDB each time
    prepare.preprocess()
    print("Process Done, exit. Please re-run this file with other flags.")
    exit(0)


if args.dataset_cache:
    training_list = open(args.training_list).readlines()
    training_list = [x.strip() for x in training_list]

    testing_list = open(args.testing_list).readlines()
    testing_list = [x.strip() for x in testing_list]

    # check whether to cache new data.
    prepare = DataPrepare(args, 
                    training_list=training_list, 
                    testing_list=testing_list)

    if os.listdir(args.dataset_path):
        if not args.dataset_override:
            raise RuntimeError(f"{args.data_path} is not empty, please clear the contents, or override it by setting --dataset_override")
        else: 
            prepare.cache()
    else:
        prepare.cache()

else:
    prepare = DataPrepare(args)


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
if args.mode == 'train':
    model = MaSIFSearch(args)

    if args.cache_model:
        print(f"cache model specified in training mode, will fine tune based on this model {args.cache_model}.")
        model.load_state_dict(torch.load(args.cache_model))

    trainer = Trainer(args=args, 
                    model=model,
                    datasets=[train_set, val_set, test_set])
    trainer.train()
    
elif args.mode == 'test':
    model = MaSIFSearch(args)

    trainer = Trainer(args=args, 
                    model=model,
                    datasets=[train_set, val_set, test_set])
    try:
        model.load_state_dict(torch.load(args.cache_model))
    except Exception as e:
        print(e)
        raise RuntimeError("Model path not (corretly) specified.")
        
    trainer.test()

else:
    raise KeyError("Running mode not supported.")