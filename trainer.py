# -*- coding: utf-8 -*-
'''
@File   :  trainer.py
@Time   :  2023/12/14 17:14
@Author :  Yufan Liu
@Desc   :  define a trainer for training and testing
'''

class Trainer(object):
    """A trainer class, for training and testing
    """
    def __init__(self, 
                 model,
                 optimizer,
                 ):
        self.optimizer = optimizer
        self.model=model

    def train(self,dataset):
        pass

    def inference(self, data_path):
        pass