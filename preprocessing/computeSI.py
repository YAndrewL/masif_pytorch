# -*- coding: utf-8 -*-
'''
@File   :  mesh_features.py
@Time   :  2023/12/11 11:23
@Author :  Yufan Liu
@Desc   :  Compute Shape Index
'''

import numpy as np
 
def generate_shapeindex(mesh):
   # Gaussian and Mean 
    #num_v = mesh.current_mesh().vertex_number()
    #mesh.meshing_repair_non_manifold_vertices()  # remove edges
    _ = mesh.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype='Mean Curvature')    
    # this will update vertex scalar
    H = mesh.current_mesh().vertex_scalar_array()
    _ = mesh.compute_scalar_by_discrete_curvature_per_vertex(curvaturetype='Gaussian Curvature')    
    K = mesh.current_mesh().vertex_scalar_array()

    elem = np.square(H) - K
    elem[elem < 0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    # Compute the shape index 
    si = (k1+k2)/(k1-k2)
    si = np.arctan(si)*(2/np.pi)
    #assert(len(si) == num_v), print(len(si), num_v)

    mesh.set_attribute('shape_index', si)

    return mesh



