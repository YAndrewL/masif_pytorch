# -*- coding: utf-8 -*-
'''
@File   :  sampler.py
@Time   :  2023/12/13 13:59
@Author :  Yufan Liu
@Desc   :  Split positive, negative and binders
'''

from tqdm import tqdm
import os
from .geomesh import GeoMesh
import numpy as np
from scipy.spatial import cKDTree

def generate_data_cache(args, path:str, dataset:list):
    # storage 
    binder_feature = []
    postive_feature = []
    negative_feature = []

    data_type = path.split('/')[-1]
    print(f"Processing {data_type} dataset...")
    for data in tqdm(dataset):
        data_path = args.processed_path
        mesh1 = GeoMesh(load=True)
        mesh1.load_new_mesh(os.path.join(data_path, data, 'p1.ply'))
        mesh1.load_feature(os.path.join(data_path, data, 'p1_input_feat.npy'))

        mesh2 = GeoMesh(load=True)
        mesh2.load_new_mesh(os.path.join(data_path, data, 'p2.ply'))
        mesh2.load_feature(os.path.join(data_path, data, 'p2_input_feat.npy'))

        labels = mesh1.get_attribute('v1_sc')[0]
        labels = np.median(labels, axis=1)
        pos_labels = np.where((labels < args.max_sc_filt) & (labels > args.min_sc_filt))[0]
        # accept all  positive labels
        if len(pos_labels) < 1:
            continue

        v1 = mesh1.vertices[pos_labels]
        v2 = mesh2.vertices

        kdt = cKDTree(v2)
        d, r = kdt.query(v1)
        # Contact points: those within a cutoff distance.
        contact_points = np.where(d < args.pos_interface_cutoff)[0]  # distance < 1
        k1 = pos_labels[contact_points]  # pos labels in P1 (binders)
        k2 = r[contact_points]  # pos labels in P2

        # For negatives, get points in v2 far from p1.
        kdt = cKDTree(v1)
        dneg, rneg = kdt.query(v2)
        k_neg2 = np.where(dneg > args.pos_interface_cutoff)[0]  # neg label
        assert len(k1) == len(k2) 
        n_pos = len(k1)
        np.random.shuffle(k_neg2)
        k_neg2 = k_neg2[:n_pos]

        rho1 = mesh1.get_attribute('rho')
        theta1 = mesh1.get_attribute('theta')
        input_feature1 = mesh1.get_attribute('input_feature')

        rho2 = mesh2.get_attribute('rho')
        theta2 = mesh2.get_attribute('theta')
        input_feature2 = mesh2.get_attribute('input_feature')   

        # postive
        # note here: for convenient storage, rho and theta are concat,
        # shape of feature: [N, max_vertex, 5+2]
        binder_f = np.concatenate([input_feature1[k1], 
                                   np.expand_dims(rho1[k1], 2), 
                                   np.expand_dims(theta1[k1], 2)], axis=2)
        pos_f = np.concatenate([input_feature2[k2], 
                                np.expand_dims(rho2[k2], 2), 
                                np.expand_dims(theta2[k2], 2)], axis=2)
        neg_f = np.concatenate([input_feature2[k_neg2], 
                                np.expand_dims(rho2[k_neg2], 2), 
                                np.expand_dims(theta2[k_neg2], 2)], axis=2)

        binder_feature.append(binder_f)
        postive_feature.append(pos_f)
        negative_feature.append(neg_f)
    
    binder_feature = np.concatenate(binder_feature, axis=0)
    postive_feature = np.concatenate(postive_feature, axis=0)
    negative_feature = np.concatenate(negative_feature, axis=0)

    # save 
    np.save(os.path.join(path, 'binder_feature.npy'), binder_feature)
    np.save(os.path.join(path, 'positive_feature.npy'), postive_feature)
    np.save(os.path.join(path, 'negative_feature.npy'), negative_feature)
    print("Done.")

    return True