# -*- coding: utf-8 -*-
'''
@File   :  alignment.py
@Time   :  2024/01/08 14:13
@Author :  Yufan Liu
@Desc   :  Alignment of two protein ply files using RANSAC
'''

import open3d as o3d
from open3d.pipelines import registration as o3d_pr
import numpy as np


    
class PatchAlignment(object):
    """
    Alignment of two patchs and transform protein file to do so.
    """
    def __init__(self, 
                 target_coord_file,
                 target_desc_file,
                 binder_coord_file,
                 binder_desc_file,
                 ransac_radius=1.0,
                 ransac_iter=2000
                 ):
        self.target_pcd = o3d.io.read_point_cloud(target_coord_file) 
        self.binder_pcd = o3d.io.read_point_cloud(binder_coord_file)

        target_feat = np.load(target_desc_file)
        self.target_feat = o3d_pr.Feature()
        self.target_feat.data = target_feat.T

        binder_feat = np.load(binder_desc_file)
        self.binder_feat = o3d_pr.Feature()
        self.binder_feat.data = binder_feat.T

        self.ransac_radius = ransac_radius
        self.ransac_iter = ransac_iter

    def align(self):
        result = o3d_pr.registration_ransac_based_on_feature_matching(
            self.binder_pcd,
            self.target_pcd,
            self.binder_feat,
            self.target_feat,
            False,
            self.ransac_radius,
            o3d_pr.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d_pr.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d_pr.CorrespondenceCheckerBasedOnDistance(2.0),
                o3d_pr.CorrespondenceCheckerBasedOnNormal(np.pi / 2),
            ],
            o3d_pr.RANSACConvergenceCriteria(self.ransac_iter, 500),
        )
        result = o3d_pr.registration_icp(source=self.binder_pcd,
                                         target=self.target_pcd,
                                         max_correspondence_distance=1.0,
                                         init=result.transformation,
                                         estimation_method=o3d_pr.TransformationEstimationPointToPlane()
                                         )
        return result
    
def transform_pdb(pdb_file):
    pass