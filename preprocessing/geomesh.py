# -*- coding: utf-8 -*-
'''
@File   :  geomesh.py
@Time   :  2023/12/11 19:10
@Author :  Yufan Liu
@Desc   :  Geometric mesh 
'''

import pymeshlab
import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler

class GeoMesh(pymeshlab.MeshSet):
    def __init__(self, **kwargs):
        super().__init__()
        # define sth. into meta data 
        if not 'vertex_matrix' in kwargs or not 'face_matrix' in kwargs:
            raise KeyError("not find essential keys vertex/face")

        
        self.original_mesh = pymeshlab.Mesh(vertex_matrix=kwargs['vertex_matrix'], face_matrix=kwargs['face_matrix'] )
        self.add_mesh(self.original_mesh)

        self.metadata = {}
        for key in kwargs:  # add features
            if key != 'vertex_matrix' and key != 'face_matrix':
                self.metadata[key] = kwargs[key]
        self.feat_norm_flag = False

    @property
    def vertices(self):
        return self.current_mesh().vertex_matrix()
    
    @property
    def faces(self):
        return self.current_mesh().face_matrix()

    @property
    def vertex_normals(self):
        return self.current_mesh().vertex_normal_matrix()
    
    def set_attribute(self, key, value):
        # to metadata
        self.metadata[key] = value

    def get_attribute(self, key):
        return self.metadata[key]

    def optimze_mesh(self):
        # todo pymeshlab seems provided some subtle function to finetune 
        #the surface, consider later
        pass

    def update_feature(self):
        # update features to a new, collapse mesh by interpolation 
        # return: metadata for a new mesh
        charge = self.metadata['charge']
        logp = self.metadata['logp']

        # see masif: assignNewChargeToMesh
        dataset = self.original_mesh.vertex_matrix()
        testset = self.current_mesh().vertex_matrix()

        new_charge = np.zeros(len(testset))
        new_logp = np.zeros(len(testset))

        num_inter = 4  # Number of interpolation features
        # Assign k old vertices to each new vertex.
        kdt = KDTree(dataset)
        dists, result = kdt.query(testset, k=num_inter)
        # Square the distances (as in the original pyflann)
        dists = np.square(dists)
        # The size of result is the same as new_vertices
        for vi_new in range(len(result)):
            vi_old = result[vi_new]
            dist_old = dists[vi_new]
            # If one vertex is right on top, ignore the rest.
            if dist_old[0] == 0.0:
                new_charge[vi_new] = charge[vi_old[0]]
                new_logp[vi_new] = logp[vi_old[0]]
                continue

            total_dist = np.sum(1 / dist_old)
            for i in range(num_inter):
                new_charge[vi_new] += (
                    charge[vi_old[i]] * (1 / dist_old[i]) / total_dist
                )      
                new_logp[vi_new] += (
                    logp[vi_old[i]] * (1 / dist_old[i]) / total_dist
                )                  
        assert len(new_charge) == len(new_logp)
        assert len(new_charge) == len(testset)
        self.metadata['charge'] = new_charge
        self.metadata['logp'] = new_logp

    def normalize_features(self):
        # feature order: shapeindex, ddc, charge, logp, apbs
        # only normalize logp and apbs, others are in -1 and 1
        feature_dict = self.metadata
        assert 'input_feature' in feature_dict, "Not all feature are processed!"
        # for logp
        # todo all in min-max ,need to talk
        logp = feature_dict['input_feature'][:, :, 3]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        logp = scaler.fit_transform(logp)

        # for apbs
        apbs = feature_dict['input_feature'][:, :, 4]
        apbs[apbs > 30] = 30
        apbs[apbs < -30] = -30
        apbs = scaler.fit_transform(apbs)

        feature_dict['input_feature'][:, :, 3] = logp        
        feature_dict['input_feature'][:, :, 4] = apbs
        self.feat_norm_flag = True


    def save_feature(self, file):
        feature = self.metadata['input_feature']
        if self.feat_norm_flag:
            # save features to npy
            np.save(file, feature)
        else:
            raise SyntaxError("Feature not normalized.")


    def save_to_point(self, file):
        # save coordinates to PDB
        pass


