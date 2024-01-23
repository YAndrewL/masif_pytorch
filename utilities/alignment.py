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
from typing import List


    
class PatchAlignment(object):
    """
    Alignment of two patchs and transform protein file to do so.
    """
    def __init__(self, 
                 target_pcd: List[np.ndarray],  # coordinate array and normal array
                 target_feat: np.ndarray,
                 binder_pcd: List[np.ndarray],
                 binder_feat: np.ndarray,
                 ransac_radius=1.0,
                 ransac_iter=100000,
                 confidence=500,
                 max_correspondence_distance=1
                 ):
        target_coord, target_normal = target_pcd
        binder_coord, binder_normal = binder_pcd
        
        self.target_pcd = o3d.geometry.PointCloud()
        self.target_pcd.points = o3d.utility.Vector3dVector(target_coord)
        self.target_pcd.normals = o3d.utility.Vector3dVector(target_normal)
        
        self.binder_pcd = o3d.geometry.PointCloud()
        self.binder_pcd.points = o3d.utility.Vector3dVector(binder_coord)
        self.binder_pcd.normals = o3d.utility.Vector3dVector(binder_normal)

        self.confidence = confidence
        self.max_correspondence_distance = max_correspondence_distance
        
        self.target_feat = o3d_pr.Feature()
        self.target_feat.data = target_feat.T

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
                o3d_pr.CorrespondenceCheckerBasedOnDistance(1.0),
                o3d_pr.CorrespondenceCheckerBasedOnNormal(np.pi / 2),
            ],
            o3d_pr.RANSACConvergenceCriteria(self.ransac_iter, self.confidence),
        )
        result = o3d_pr.registration_icp(source=self.binder_pcd,
                                         target=self.target_pcd,
                                         max_correspondence_distance=self.max_correspondence_distance,
                                         init=result.transformation,
                                         estimation_method=o3d_pr.TransformationEstimationPointToPlane()
                                         )
        
        # test
        #o3d.io.write_point_cloud("./target.ply", self.target_pcd, write_ascii=True)
        #o3d.io.write_point_cloud("./binder.ply", self.binder_pcd, write_ascii=True)
        #self.binder_pcd.transform(result.transformation)
        #o3d.io.write_point_cloud("./binder_after.ply", self.binder_pcd , write_ascii=True)
        
        
        return result
    