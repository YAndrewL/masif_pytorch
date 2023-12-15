# -*- coding: utf-8 -*-
'''
@File   :  main.py
@Time   :  2023/12/07 14:52
@Author :  Yufan Liu
@Desc   :  Main function for training/inference
'''
from arguments import parser
from data_prepare import DataPrepare
from dataset.dataset import SurfaceDataset, collate
from torch.utils.data import DataLoader

args = parser.parse_args()

# prepare = DataPrepare(args, training_list=['1AKJ_AB_DE', '1A0G_A_B'], 
#                       testing_list=['1A2A_C_D'])
# #status = prepare.preprocess()
# #if status:
# prepare.cache()

# dataset = SurfaceDataset(args, 'train', False)
# dataset = DataLoader(dataset, batch_size=4, collate_fn=collate)
# for data in dataset:
#     print(data)