# -*- coding: utf-8 -*-
'''
@File   :  sampler.py
@Time   :  2023/12/13 13:59
@Author :  Yufan Liu
@Desc   :  Split positive, negative and binders
'''


import os
from .geomesh import GeoMesh

def generate_data_cache(args, path:str, dataset:list):

    # storage 






    for data in dataset:
        data_path = os.path.join(path, data)
        mesh1 = GeoMesh(load=True)
        mesh1.load_new_mesh(data_path + '/.p1.ply')
        mesh1.load_feature(data_path + '/p1.input_feat.npy')

        mesh2 = GeoMesh(load=True)
        mesh2.load_new_mesh(data_path + '/.p1.ply')
        mesh2.load_feature(data_path + '/p1.input_feat.npy')


