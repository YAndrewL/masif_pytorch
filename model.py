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
                  n_rotations,
                  n_features):
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
        self.n_features = n_features
        
        # n_gauss = n_rhos * n_thetas
        initial_coords = self.compute_initial_coordinate() 
        mu_rho_initial = initial_coords[:, 0].unsqueeze(0)  # [1, n_gauss]
        mu_theta_initial = initial_coords[:, 1].unsqueeze(0)
        self.mu_rho = []
        self.mu_theta = []
        self.sigma_rho = []
        self.sigma_theta = []
        # todo I dont know why initilized as this, since the set of parameters are trainable
        for i in range(self.n_features):
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
        wconv = []
        for i in range(self.n_features):
            self.b_conv.append(
                nn.Parameter(torch.Tensor(self.n_thetas * self.n_rhos)).to(self.device)
            )
            wconv.append(nn.Parameter(torch.Tensor(self.n_thetas * self.n_rhos,
                                                self.n_thetas * self.n_rhos)).to(self.device)
            ) 
        self.W_conv = [nn.init.xavier_normal(wconv[i]) for i in range(self.n_features)]


    def compute_initial_coordinate(self):
        # initialize a polar mesh grid
        range_rho = torch.linspace(0.0, self.max_rho, steps=self.n_rhos + 1)
        range_theta = torch.linspace(0, 2 * np.pi, steps=self.n_thetas + 1)

        # rho!=0 and theta!=2pi
        range_rho = range_rho[1:]
        range_theta = range_theta[:-1]

        # Creating the mesh grid using torch.meshgrid
        grid_rho, grid_theta = torch.meshgrid(range_rho, range_theta, indexing='ij')
        # Flattening the grid arrays
        grid_rho = grid_rho.flatten()
        grid_theta = grid_theta.flatten()

        # Combining the coordinates
        coords = torch.stack((grid_rho, grid_theta), dim=1)

        return coords  # [n_rhos * n_thetas, 2]

    def gauss_conv(self, 
                input_feat, 
                rho, 
                theta,
                mu_rho,
                mu_theta,
                sigma_rho,
                sigma_theta,
                W_conv,
                b_conv,
                eps=1e-5,
                mean_gauss_activation=True):
        """
        feature: [batch_size, n_vertex, 5]
        rho and theta: [batch_size, n_vertex, 1]
        """
        n_samples = rho.size(0)
        n_vertice = rho.size(1)

        all_conv_feat = []
        for k in range(self.n_rotations):  # against coordinate ambiguity
            rho_coords_ = rho.reshape(-1, 1)  
            theta_coords_ = theta.reshape(-1, 1)
            theta_coords_ += k * 2 * pi / self.n_rotations
            theta_coords_ = torch.remainder(theta_coords_, 2 * pi)

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
            conv_feat = torch.matmul(gauss_desc, W_conv) + b_conv 

            all_conv_feat.append(conv_feat)
        all_conv_feat = torch.stack(all_conv_feat)
        conv_feat = torch.max(all_conv_feat, 0)
        out = conv_feat.values  # [batch_size, n_rhos * n_thetas]
        return out

    def forward(self, feature, rhos, thetas):
        all_desc = []
        for i in range(self.n_features):
            desc = self.gauss_conv(input_feat=feature[:, :, i].unsqueeze(2),  # [N, V, 1]
                                   rho=rhos, 
                                   theta=thetas,
                                   mu_rho=self.mu_rho[i],
                                   mu_theta=self.mu_theta[i],
                                   sigma_rho=self.sigma_rho[i],
                                   sigma_theta=self.sigma_theta[i],
                                   W_conv=self.W_conv[i],
                                   b_conv=self.b_conv[i])
            all_desc.append(desc)
        all_desc = torch.stack(all_desc, dim=0)  # [n_feat, N, dim]  # lyf stack axis should be zero!
        
        return all_desc.reshape(-1, self.n_features * self.n_rhos * self.n_thetas)

class MaSIFSearch(nn.Module):

    """
    LYF Note:
    Take MoNet as a main reference: https://arxiv.org/abs/1611.08402 (Equation 9)
    For each dimension of feature, a pseudo-coordinate is defined (Table1), where we used acutal
    polar coordinate in this model. Further, use weight function (filter) to generate descriptor for 
    each point (vertex) using coordinates respect to neighbor. Filter is Gaussian as Equation 11.
    """

    def __init__(self,
        args,
        # ignore the feature mask for albation, do if needed
        ):

        super().__init__()
        self.args = args
        self.device = args.device
        self.n_thetas = args.n_thetas
        self.n_rhos = args.n_rhos
        self.n_features = args.n_features
        self.n_rotations = args.n_rotations

        self.gauss_conv = GaussianFiler(args=args, 
                                        n_thetas=self.n_thetas, 
                                        n_rhos=self.n_rhos, 
                                        n_rotations=self.n_rotations,
                                        n_features=self.n_features)
        self.fcc = nn.Linear(self.n_thetas * self.n_rhos * self.n_features,
                             self.n_thetas * self.n_rhos)
        self.relu = nn.ReLU()


    def forward(self, batch):
        desc = []
        for data in batch:
            data = data.to(self.args.device)
            feature = data[:, :, :5]
            rho = data[:, :, 5:6]
            theta = data[:,:,6:7]
            out = self.gauss_conv(feature=feature, rhos=rho, thetas=theta)
            out = self.relu(out)
            out = self.fcc(out)
            desc.append(out)
        return desc

        




        



