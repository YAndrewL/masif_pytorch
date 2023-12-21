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
from tqdm import tqdm
import yaml


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
        self.model_path = os.path.join(args.model_path, args.experiment_name, timestamp)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path, exist_ok=True)

        self.logger = logger.add(os.path.join(self.model_path, 'logger.log'))
        self.savefile = os.path.join(self.model_path, 'model.pth')
        
        # save config
        config_file = os.path.join(self.model_path, 'config.yaml')
        self.config_save(config_file)

        self.relu = nn.ReLU()

        # parameters
        for name, params in self.model.named_parameters():
            logger.info(f"Trainable parameters in model: {name}")
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Amount of trainable parameters : {n_params}")


    def train(self):
        #torch.autograd.set_detect_anomaly(True)
        logger.info("Training start!")
        best_val_auc = -1 
        for epoch in range(self.epochs):
            self.model.train()
            train_score = []
            pbar = tqdm(self.train_set)
            t_loss = []
            t_score = []
            for data in pbar:
                outputs = self.model(data)
                loss, score = self.compute_loss(outputs)

                # if torch.isnan(loss):
                #     print(data)
                #     d=[_.detach().cpu().numpy() for _ in data]
                #     for i,v in enumerate(d):
                #         np.save(f"{i}.npy", v)
                #     print([_.sum() for _ in data])
                #     exit()
                
                pbar.set_postfix({"train loss": loss.item(), 
                                  "positive mean": score[0].mean().item(),
                                  "negative mean": score[1].mean().item()})
                self.optimizer.zero_grad()
                loss.backward()

                # for p in self.model.parameters():
                #     if p.grad is not None and torch.isnan(p.grad.sum()):
                #         print(p.shape, loss, score)
                #         exit()
                self.optimizer.step()
                train_score.append(score)  
                t_loss.append(loss)
                t_score.append(score)


            pos = torch.cat([d[0] for d in t_score])  # [N-sample,]
            neg = torch.cat([d[1] for d in t_score])     
            roc = 1 - self.compute_roc_auc(pos, neg)

            logger.info(f"Epoch {epoch} / {self.epochs}, Training loss: {torch.mean(torch.stack(t_loss)).item():.6f}")
            logger.info(f"Training AUR-ROC: {roc.item():.6f}")


            if epoch % self.args.test_epochs == 0:
                logger.warning(f"Iteration reached, validate and test!")
                
                # validtion
                loss = []
                score = []
                self.model.eval()
                with torch.no_grad():
                    for data in tqdm(self.val_set):

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

                    # for name, params in self.model.named_parameters():
                    #     if 'b_conv' in name:
                    #         print(params)

                    if roc.item() > best_val_auc:
                        torch.save(self.model.state_dict(),
                                self.savefile)
                        best_val_auc = roc.item()
                        logger.critical("Better validation AUC, model saved.")

                    loss = []
                    score = []
                    for data in tqdm(self.test_set):
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

    def compute_roc_auc(self, pos, neg):
        pos = pos.detach().cpu().numpy()
        neg = neg.detach().cpu().numpy()
        labels = np.concatenate([np.ones((len(pos))), np.zeros((len(neg)))])
        dist_pairs = np.concatenate([pos, neg])
        return roc_auc_score(labels, dist_pairs)   

    def compute_loss(self, outputs):
        # descriptos
        binder, pos, neg = outputs
        dist_p = self.dist(binder, pos)
        dist_n = self.dist(neg, binder)

        pos_distance = self.relu(dist_p - self.args.pos_thresh)
        neg_distance = self.relu(-dist_n + self.args.neg_thresh)

        score = (dist_p, dist_n)
        pos_mean, pos_std = torch.mean(pos_distance, 0), torch.var(pos_distance, 0)
        neg_mean, neg_std = torch.mean(neg_distance, 0), torch.var(neg_distance, 0)
        # print(pos_mean, neg_mean)
        # print(score)
        loss = pos_mean + pos_std + neg_mean + neg_std
        return loss, score

    def dist(self, a, b):
        assert a.shape == b.shape
        return torch.sum(torch.square(a - b), 1)
    
    def config_save(self, saver):
        save_dict = vars(self.args)
        assert saver.split('.')[-1] == 'yaml'
        with open(saver, 'w') as file:
            yaml.dump(save_dict, file)