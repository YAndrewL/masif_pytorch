# -*- coding: utf-8 -*-
'''
@File   :  computeDDC.py
@Time   :  2023/12/12 17:26
@Author :  Yufan Liu
@Desc   :  Compute distance dependent curvature
Credit to PGC MaSIF
'''


import numpy as np

def generate_ddc(mesh, max_shape=200):

    n = len(mesh.vertices)
    neigh_indices = mesh.get_attribute('neighbor_id')
    rho = mesh.get_attribute('rho')
    normals = mesh.vertex_normals
    input_feat = np.zeros((n, max_shape, 5))

    shapeindex = mesh.get_attribute('shape_index')
    charge = mesh.get_attribute('charge')
    logp = mesh.get_attribute('logp')
    apbs = mesh.get_attribute('apbs')
    mask = mesh.get_attribute('neighbor_mask')

    for vix in range(n):
        neigh_vix = np.array(neigh_indices[vix])
        # Compute the distance-dependent curvature for all neighbors of the patch. 
        patch_v = mesh.vertices[neigh_vix]
        patch_n = normals[neigh_vix]
        patch_cp = np.where(neigh_vix == vix)[0][0] # central point
        mask_pos = np.where(mask[vix] == 1.0)[0] # nonzero elements
        patch_rho = rho[vix][mask_pos]
        ddc = compute_ddc(patch_v, patch_n, patch_cp, patch_rho)        
        
        input_feat[vix, :len(neigh_vix), 0] = shapeindex[neigh_vix]
        input_feat[vix, :len(neigh_vix), 1] = ddc
        input_feat[vix, :len(neigh_vix), 2] = charge[neigh_vix]
        input_feat[vix, :len(neigh_vix), 3] = logp[neigh_vix]
        input_feat[vix, :len(neigh_vix), 4] = apbs[neigh_vix]
    
    mesh.set_attribute('input_feature', input_feat)
    return mesh


def compute_ddc(patch_v, patch_n, patch_cp, patch_rho):
    """
        Compute the distance dependent curvature, Yin et al PNAS 2009
            patch_v: the patch vertices
            patch_n: the patch normals
            patch_cp: the index of the central point of the patch 
            patch_rho: the geodesic distance to all members.
        Returns a vector with the ddc for each point in the patch.
    """
    n = patch_n
    r = patch_v
    i = patch_cp
    # Compute the mean normal 2.5A around the center point
    ni = mean_normal_center_patch(patch_rho, n, 2.5)
    dij = np.linalg.norm(r - r[i], axis=1)
    # Compute the step function sf:
    sf = r + n
    sf = sf - (ni + r[i])
    sf = np.linalg.norm(sf, axis=1)
    sf = sf - dij
    sf[sf > 0] = 1
    sf[sf < 0] = -1
    sf[sf == 0] = 0
    # Compute the curvature between i and j
    dij[dij == 0] = 1e-8
    kij = np.divide(np.linalg.norm(n - ni, axis=1), dij)
    kij = np.multiply(sf, kij)
    # Ignore any values greater than 0.7 and any values smaller than 0.7
    kij[kij > 0.7] = 0
    kij[kij < -0.7] = 0
    return kij


def mean_normal_center_patch(D, n, r):
    """
        Function to compute the mean normal of vertices within r radius of the center of the patch.
    """
    c_normal = [n[i] for i in range(len(D)) if D[i] <= r]
    mean_normal = np.mean(c_normal, axis=0, keepdims=True).T
    mean_normal = mean_normal / np.linalg.norm(mean_normal)
    return np.squeeze(mean_normal)