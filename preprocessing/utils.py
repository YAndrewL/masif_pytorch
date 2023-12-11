# -*- coding: utf-8 -*-
'''
@File   :  utils.py
@Time   :  2023/12/10 20:09
@Author :  Yufan Liu
@Desc   :  Some helper functions
'''

# I defined a new class here to fit pymesh, and store/output features
from trimesh import Trimesh

class GeoMesh(Trimesh):
    def __init__(self):
        super().__init__()
        # define sth. into meta data    

    def add_attribute(self, key, value):
        # to metadata
        self.metadata[key] = value

    def get_attribute(self, key):
        return self.metadata[key]

    def set_attribute(self, key):
        # looks nonmeaningful?
        self.metadata[key] = ''


    def save_to(self, file):
        # particularly save to ply with numerical without detail check. I'm not writing a package
        # properties [x, y, z, nx, ny, nz]
        # features [charge, logp, apbs_charge, shapeindex, ddc]
        pass





