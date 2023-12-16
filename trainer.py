# -*- coding: utf-8 -*-
'''
@File   :  trainer.py
@Time   :  2023/12/14 17:14
@Author :  Yufan Liu
@Desc   :  define a trainer for training and testing
'''

import torch
import torch.nn as nn
from loguru import logger


class Trainer(object):
    """A trainer class, for training and testing
    """
    def __init__(self, 
                 args,
                 model,
                 datasets:list,  # receive datalist
                 ):
        self.args = args
        self.epochs = args.epochs
        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        self.model = model
        self.train_set, self.val_set, self.test_set = datasets


    def train(self):
        for epoch in range(self.epochs):
            
        




    def loss_computation(self):
        pass