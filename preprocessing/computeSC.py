# -*- coding: utf-8 -*-
'''
@File   :  computeSC.py
@Time   :  2023/12/14 10:24
@Author :  Yufan Liu
@Desc   :  Compute shape complementarity
'''
import numpy as np
from scipy.spatial import cKDTree

def generate_shape_complementarity(args, meshes):
    mesh1, mesh2 = meshes
    n1 = mesh1.vertex_normals
    n2 = mesh2.vertex_normals
    v1 = mesh1.vertices 
    v2 = mesh2.vertices
    mask1 = mesh1.metadata['neighbor_mask']
    mask2 = mesh2.metadata['neighbor_mask']
    neigh1 = mesh1.metadata['neighbor_mask']
    neigh2 = mesh2.metadata['neighbor_mask']
    rho1 = mesh1.metadata['rho']
    rho2 = mesh2.metadata['rho']



    w = args.sc_w
    int_cutoff = args.sc_interaction_cutoff  # less than 1.5 are defined as ppi
    radius = args.sc_radius

    num_rings = 10
    scales = np.arange(0, radius, radius/10)
    scales = np.append(scales, radius)



    v1_sc = np.zeros((2,len(v1), 10))
    v2_sc = np.zeros((2,len(v2), 10))

    # Find all interface vertices
    kdt = cKDTree(v2)
    d, nearest_neighbors_v1_to_v2 = kdt.query(v1)
    # Interface vertices in v1
    interface_vertices_v1 = np.where(d < int_cutoff)[0]

    # Go through every interface vertex. 
    for cv1_iiix in range(len(interface_vertices_v1)):
        cv1_ix = interface_vertices_v1[cv1_iiix]
        assert (d[cv1_ix] < int_cutoff)
        # First shape complementarity s1->s2 for the entire patch
        patch_idxs1 = np.where(mask1[cv1_ix]==1)[0]
        neigh_cv1 = np.array(neigh1[cv1_ix])[patch_idxs1]
        # Find the point cv2_ix in s2 that is closest to cv1_ix
        cv2_ix = nearest_neighbors_v1_to_v2[cv1_ix]
        patch_idxs2 = np.where(mask2[cv2_ix]==1)[0]
        neigh_cv2 = np.array(neigh2[cv2_ix])[patch_idxs2]

        patch_v1 = v1[neigh_cv1]
        patch_v2 = v2[neigh_cv2]
        patch_n1 = n1[neigh_cv1]  #  compute by normal
        patch_n2 = n2[neigh_cv2]

        patch_kdt = cKDTree(patch_v1)
        p_dists_v2_to_v1, p_nearest_neighbor_v2_to_v1 = patch_kdt.query(patch_v2)
        patch_kdt = cKDTree(patch_v2)
        p_dists_v1_to_v2, p_nearest_neighbor_v1_to_v2 = patch_kdt.query(patch_v1)
        
        # First v1->v2
        neigh_cv1_p = p_nearest_neighbor_v1_to_v2
        comp1 = [np.dot(patch_n1[x], -patch_n2[neigh_cv1_p][x]) for x in range(len(patch_n1))]
        comp1 = np.multiply(comp1, np.exp(-w * np.square(p_dists_v1_to_v2)))
        # Use 10 rings such that each ring has equal weight in shape complementarity
        comp_rings1_25 = np.zeros(num_rings)
        comp_rings1_50 = np.zeros(num_rings)

        patch_rho1 = np.array(rho1[cv1_ix])[patch_idxs1]
        for ring in range(num_rings):
            scale = scales[ring]
            members = np.where((patch_rho1 >= scales[ring]) & (patch_rho1 < scales[ring + 1]))
            if len(members[0]) == 0:
                comp_rings1_25[ring] = 0.0
                comp_rings1_50[ring] = 0.0
            else:
                comp_rings1_25[ring] = np.percentile(comp1[members], 25)
                comp_rings1_50[ring] = np.percentile(comp1[members], 50)
        
        # Now v2->v1
        neigh_cv2_p = p_nearest_neighbor_v2_to_v1
        comp2 = [np.dot(patch_n2[x], -patch_n1[neigh_cv2_p][x]) for x in range(len(patch_n2))]
        comp2 = np.multiply(comp2, np.exp(-w * np.square(p_dists_v2_to_v1)))
        # Use 10 rings such that each ring has equal weight in shape complementarity
        comp_rings2_25 = np.zeros(num_rings)
        comp_rings2_50 = np.zeros(num_rings)

        # Apply mask to patch rho coordinates. 
        patch_rho2 = np.array(rho2[cv2_ix])[patch_idxs2]
        for ring in range(num_rings):
            scale = scales[ring]
            members = np.where((patch_rho2 >= scales[ring]) & (patch_rho2 < scales[ring + 1]))
            if len(members[0]) == 0:
                comp_rings2_25[ring] = 0.0
                comp_rings2_50[ring] = 0.0
            else:
                comp_rings2_25[ring] = np.percentile(comp2[members], 25)
                comp_rings2_50[ring] = np.percentile(comp2[members], 50)

        v1_sc[0,cv1_ix,:] = comp_rings1_25
        v2_sc[0,cv2_ix,:] = comp_rings2_25
        v1_sc[1,cv1_ix,:] = comp_rings1_50
        v2_sc[1,cv2_ix,:] = comp_rings2_50

    mesh1.set_attribute('v1_sc', v1_sc)
    mesh2.set_attribute('v2_sc', v2_sc)

    return mesh1, mesh2


