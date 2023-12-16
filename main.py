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
prepare = DataPrepare(args, training_list=['1AKJ_AB_DE', '1A0G_A_B', '3I71_A_B', '3IA0_I_J'], 
                      testing_list=['1A2A_C_D', '3IDF_A_B'])
# prepare.preprocess()
#prepare.cache()

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
