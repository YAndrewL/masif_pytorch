# -*- coding: utf-8 -*-
'''
@File   :  model.py
@Time   :  2023/12/14 13:12
@Author :  Yufan Liu
@Desc   :  MaSIF model by Pytorch
'''


import torch  
import torch.nn as nn
import numpy as np
pi = torch.tensor(3.141592653589793)

class GaussianFiler(nn.Module):
    """
    Define a Gaussian filter to conduct per-feature pattern matching
    """
    def  __init__(self,
                  args,
                  n_thetas,
                  n_rhos,
                  n_rotations):
        super().__init__()
        self.max_rho = args.max_distance
        self.n_thetas = n_thetas
        self.n_rhos  = n_rhos
        self.n_rotations = n_rotations  
        # note: from paper: 
        # take all possible rotations (16 here) and take max as final,
        # to resolve the origin ambiguity in angular coordinate
        self.device = args.device

        self.sigma_rho_init = (
            self.max_rho / 8
        )  # in MoNet was 0.005 with max radius=0.04 (i.e. 8 times smaller)
        self.sigma_theta_init = 1.0  # 0.25
        self.n_rotations = n_rotations
        self.n_feat = 5
        
        initial_coords = self.compute_initial_coordinates() 
        mu_rho_initial = initial_coords[:, 0].unsqueeze(0)  # [n_rho * n_theta, 1]
        mu_theta_initial = initial_coords[:, 1].unsqueeze(0)
        self.mu_rho = []
        self.mu_theta = []
        self.sigma_rho = []
        self.sigma_theta = []
        # todo I dont know why initilized as this, since the set of parameters are trainable
        print(mu_rho_initial.shape)
        for i in range(self.n_feat):
            self.mu_rho.append(
                nn.Parameter(mu_rho_initial).to(self.device)
            )  # 1, n_gauss
            self.mu_theta.append(
                nn.Parameter(mu_theta_initial).to(self.device)
            )  # 1, n_gauss
            self.sigma_rho.append(
                nn.Parameter(torch.ones_like(mu_rho_initial) * self.sigma_rho_init).to(self.device)
            )  # 1, n_gauss
            self.sigma_theta.append(
                nn.Parameter(torch.ones_like(mu_theta_initial) * self.sigma_theta_init).to(self.device)
            )  # 1, n_gauss

        self.global_desc = []
        # geodisc conv, trainable
        self.b_conv = []
        for i in range(self.n_feat):
            self.b_conv.append(
                nn.Parameter(torch.FloatTensor(self.n_thetas * self.n_rhos)).double().to(self.device)
            )
        # for i in range(self.n_feat):
        self.W_conv = nn.Parameter(torch.FloatTensor(self.n_thetas * self.n_rhos,
                                            self.n_thetas * self.n_rhos)).double().to(self.device) # to float 64  # todo  change abit
        nn.init.xavier_normal_(self.W_conv)




    def compute_initial_coordinate(self):
        # initialize a polar mesh grid
        range_rho = torch.linspace(0.0, self.max_rho, steps=self.n_rhos + 1)
        range_theta = torch.linspace(0, 2 * np.pi, steps=self.n_thetas + 1)

        # rho!=0 and theta!=2pi
        range_rho = range_rho[1:]
        range_theta = range_theta[:-1]

        # Creating the mesh grid using torch.meshgrid
        grid_rho, grid_theta = torch.meshgrid(range_rho, range_theta, indexing='ij')
        print(grid_rho)
        # Flattening the grid arrays
        grid_rho = grid_rho.flatten()
        grid_theta = grid_theta.flatten()

        # Combining the coordinates
        coords = torch.stack((grid_rho, grid_theta), dim=1)

        return coords  # [n_rhos * n_thetas, 2]

    def forward(self, feature, rho, theta):
        pass











class MaSIF_ppi_search(nn.Module):

    """
    LYF Note:
    Take MoNet as a main reference: https://arxiv.org/abs/1611.08402 (Equation 9)
    For each dimension of feature, a pseudo-coordinate is defined (Table1), where we used acutal
    polar coordinate in this model. Further, use weight function (filter) to generate descriptor for 
    each point (vertex) using coordinates respect to neighbor. Filter is Gaussian as Equation 11.
    """

    def __init__(self,
        args,
        max_rho,
        n_thetas=16,
        n_rhos=5,
        n_rotations=16,
        # ignore the feature mask for albation, do if needed
        ):

        super().__init__()
        self.device = args.device
        self.max_rho = max_rho
        self.n_thetas = n_thetas
        self.n_rhos = n_rhos

 # for ablation I think

        # initialize polar coords, contruct Gaussian kernel
        initial_coords = self.compute_initial_coordinates()
        mu_rho_initial = initial_coords[:, 0].unsqueeze(0)
        mu_theta_initial = initial_coords[:, 1].unsqueeze(0)
        self.mu_rho = []
        self.mu_theta = []
        self.sigma_rho = []
        self.sigma_theta = []
        # todo seems to be trainable? # todo device is not solved
        print(mu_rho_initial.shape)
        for i in range(self.n_feat):
            self.mu_rho.append(
                mu_rho_initial
            )  # 1, n_gauss
            self.mu_theta.append(
                mu_theta_initial
            )  # 1, n_gauss
            self.sigma_rho.append(
                mu_rho_initial * self.sigma_rho_init
            )  # 1, n_gauss
            self.sigma_theta.append(
                mu_theta_initial * self.sigma_theta_init
            )  # 1, n_gauss

        self.global_desc = []
        # geodisc conv, trainable
        self.b_conv = []
        for i in range(self.n_feat):
            self.b_conv.append(
                nn.Parameter(torch.FloatTensor(self.n_thetas * self.n_rhos)).double().to(self.device)
            )
        # for i in range(self.n_feat):
        self.W_conv = nn.Parameter(torch.FloatTensor(self.n_thetas * self.n_rhos,
                                            self.n_thetas * self.n_rhos)).double().to(self.device) # to float 64  # todo  change abit
        nn.init.xavier_normal_(self.W_conv)

        self.fcc = nn.Linear(self.n_thetas * self.n_rhos * self.n_feat,
                             self.n_thetas * self.n_rhos)
        self.relu = nn.ReLU()

    def forward(self,
                batch):
        # forward need to calculate descriptor and loss is computed in
        # pth_MaSIF_ppi_search.py
        self.global_desc = []
        for i in range(self.n_feat):
            my_input_feat = input_feat[:, :, i].unsqueeze(2)
            # for each feature do the conv, conv weight is the same in paper
            desc = self.inference(
                my_input_feat,
                rho_coords,
                theta_coords,
                mask,
                self.W_conv,
                self.b_conv[i],
                self.mu_rho[i],
                self.sigma_rho[i],
                self.mu_theta[i],
                self.sigma_theta[i],
            ) 
            # print(desc)
            desc = self.relu(desc)
            self.global_desc.append(desc)
        self.global_desc = torch.cat(self.global_desc, dim=1)
        self.global_desc = self.global_desc.reshape(-1, 
                                                    self.n_thetas * self.n_rhos * self.n_feat).float()

        self.global_desc = self.fcc(self.global_desc)
        self.n_patches = self.global_desc.shape[0] // 4  # lyf why, this will be devided as four parts of loss, check data laoding
        self.data_loss = self.compute_data_loss()
        return self.global_desc, self.data_loss, self.score


    def compute_data_loss(self, pos_thresh=0.0, neg_thresh=10):
        # gather is not act the same way as pth, # todo 
        print(self.global_desc.shape)
        self.global_desc_pos = self.global_desc[torch.arange(0, self.n_patches)]
        self.global_desc_binder = self.global_desc[torch.arange(self.n_patches, 2 * self.n_patches)]

        self.global_desc_neg = self.global_desc[torch.arange(2 * self.n_patches, 3 * self.n_patches)]
        self.global_desc_neg_2 = self.global_desc[torch.arange(3 * self.n_patches, 4 * self.n_patches)]

        pos_distances = torch.sum(
            torch.square(self.global_desc_binder - self.global_desc_pos), 1
        )
        neg_distances = torch.sum(
            torch.square(self.global_desc_neg - self.global_desc_neg_2), 1
        )
        self.score = torch.cat([pos_distances, neg_distances], axis=0)
        
        pos_distances = torch.sum(
            torch.square(self.global_desc_binder - self.global_desc_pos), 1
        )
        pos_distances = self.relu(pos_distances - pos_thresh)

        neg_distances = torch.sum(
            torch.square(self.global_desc_neg - self.global_desc_neg_2), 1
        )
        neg_distances = self.relu(-neg_distances + neg_thresh)

        pos_mean, pos_std = torch.mean(pos_distances, 0), torch.std(pos_distances, 0)
        neg_mean, neg_std = torch.mean(neg_distances, 0), torch.std(neg_distances, 0)
        data_loss = pos_std + neg_std + pos_mean + neg_mean

        return data_loss        
        
    
    def inference(
        self,
        input_feat,
        rho_coords,
        theta_coords,
        mask,
        W_conv,
        b_conv,
        mu_rho,
        sigma_rho,
        mu_theta,
        sigma_theta,
        eps=1e-5,
        mean_gauss_activation=True,
    ):
    
        n_samples = rho_coords.size(0)
        n_vertice = rho_coords.size(1)

        all_conv_feat = []
        for k in range(self.n_rotations):  # what is this for?
            rho_coords_ = rho_coords.reshape(-1, 1)
            theta_coords_ = theta_coords.reshape(-1, 1)
            theta_coords_ += k * 2 * pi / self.n_rotations
            theta_coords_ = torch.remainder(theta_coords_, 2 * pi)
            # todo I guess rho and theta should be learnable
            rho_coords_ = torch.exp(
                -torch.square(rho_coords_ - mu_rho) / (torch.square(sigma_rho) + eps)
            )
            theta_coords_ = torch.exp(
                -torch.square(theta_coords_ - mu_theta) / (torch.square(sigma_theta) + eps)
            )

            gauss_activations = torch.mul(rho_coords_, theta_coords_)
            gauss_activations = gauss_activations.reshape(
                n_samples, n_vertice, -1
            ) # batch_size, n_vertices, n_gauss
            gauss_activations = torch.mul(gauss_activations, mask)
            if (
                mean_gauss_activation
            ):  # computes mean weights for the different gaussians
                gauss_activations /= (
                    torch.sum(gauss_activations, dim=1, keepdim=True) + eps
                )  # batch_size, n_vertices, n_gauss

            gauss_activations = gauss_activations.unsqueeze(2)  
            # batch_size, n_vertices, 1, n_gauss,
            input_feat_ = input_feat.unsqueeze(3)
            # batch_size, n_vertices, n_feat, 1

            gauss_desc = torch.mul(
                gauss_activations, input_feat_
            )  # batch_size, n_vertices, n_feat, n_gauss,

            gauss_desc = torch.sum(gauss_desc, dim=1)  # batch_size, n_feat, n_gauss,
            gauss_desc = gauss_desc.reshape(n_samples, self.n_thetas * self.n_rhos)
            conv_feat = torch.matmul(gauss_desc, W_conv) + b_conv  # batch_size, 80
            all_conv_feat.append(conv_feat)
        all_conv_feat = torch.stack(all_conv_feat)
        conv_feat = torch.max(all_conv_feat, 0)
        out = conv_feat.values

        # conv_feat = tf.nn.relu(conv_feat)
        return out




# some functions do not need backbpropagation
    