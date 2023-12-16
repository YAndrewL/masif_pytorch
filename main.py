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

args = parser.parse_args()

# prepare data
# note: execute prepare.preprocess in advance due to speed issue
training_list = open("data/list/train_all.txt").readlines()
training_list = [x.strip() for x in training_list]

testing_list = open("data/list/test_all.txt").readlines()
testing_list = [x.strip() for x in testing_list]

prepare = DataPrepare(args, 
                      training_list=training_list, 
                      testing_list=testing_list)
# prepare.preprocess()
# prepare.cache()

# dataset
train_set = prepare.dataset(data_type='train',
                            batch_size=args.batch_size,
                            pair_shuffle=False)
val_set = prepare.dataset(data_type='val',
                            batch_size=args.batch_size,
                            pair_shuffle=False)
test_set = prepare.dataset(data_type='test',
                            batch_size=args.batch_size,
                            pair_shuffle=False)

# essential part for training
model = MaSIFSearch(args)

trainer = Trainer(args=args, 
                  model=model,
                  datasets=[train_set, val_set, test_set])
trainer.train()
