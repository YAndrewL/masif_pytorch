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
import datetime
import os
import numpy as np
from sklearn.metrics import roc_auc_score


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
        self.model = model.to(args.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=args.learning_rate)
        self.train_set, self.val_set, self.test_set = datasets
        
        timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
        self.model_path = os.path.join(args.model_path, timestamp)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)

        self.logger = logger.add(os.path.join(self.model_path, 'logger.log'))
        self.savefile = os.path.join(self.model_path, 'model.pth')
        self.relu = nn.ReLU()

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        logger.info("Training start!")
        for epoch in range(self.epochs):
            self.model.train()
            train_score = []
            for data in self.train_set:
                outputs = self.model(data)
                loss, score = self.compute_loss(outputs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_score.append(score)  # todo why score is computed as 1/dist


            if epoch % self.args.test_epochs == 0:
                logger.success(f"Epoch: {epoch}/{self.epochs}")
                logger.info(f"Training loss: {loss.item():.6f}")
                
                # validtion
                loss = []
                score = []
                for data in self.val_set:
                    self.model.eval()
                    outputs = self.model(data)
                    loss_, score_ = self.compute_loss(outputs)
                    loss.append(loss_)
                    score.append(score_)
                loss = torch.mean(torch.stack(loss))
                # score: [(pos:, neg), ...]
                pos = torch.cat([d[0] for d in score])  # [N-sample,]
                neg = torch.cat([d[1] for d in score])
                roc = 1 - self.compute_roc_auc(pos, neg)
                logger.info(f"Validating loss: {loss.item():.6f}")
                logger.info(f"Validating AUR-ROC: {roc.item():.6f}")

                loss = []
                score = []
                for data in self.test_set:
                    self.model.eval()
                    outputs = self.model(data)
                    loss_, score_ = self.compute_loss(outputs)
                    loss.append(loss_)
                    score.append(score_)
                loss = torch.mean(torch.stack(loss))
                # score: [(pos:, neg), ...]
                pos = torch.cat([d[0] for d in score])  # [N-sample,]
                neg = torch.cat([d[1] for d in score])
                roc = 1 - self.compute_roc_auc(pos, neg)
                logger.info(f"testing loss: {loss.item():.6f}")
                logger.info(f"testing AUR-ROC: {roc.item():.6f}")

    def compute_roc_auc(pos, neg):
        pos = pos.detach().cpu().numpy()
        neg = neg.detach().cpu().numpy()
        labels = np.concatenate([np.ones((len(pos))), np.zeros((len(neg)))])
        dist_pairs = np.concatenate([pos, neg])
        return roc_auc_score(labels, dist_pairs)   

    def compute_loss(self, outputs):
        # descriptos
        binder, pos, neg = outputs
        pos_distance = self.relu(self.dist(binder, pos))
        neg_distance = self.relu(self.dist(neg, binder))

        score = (pos_distance, neg_distance)
        pos_mean, pos_std = torch.mean(pos_distance, 0), torch.std(pos_distance, 0)
        neg_mean, neg_std = torch.mean(neg_distance, 0), torch.std(neg_distance, 0)
        loss = pos_mean + pos_std + neg_mean + neg_std
        return loss, score

    def dist(self, a, b):
        return torch.sum(torch.square(a - b), 1)