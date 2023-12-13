# -*- coding: utf-8 -*-
'''
@File   :  data_prepare.py
@Time   :  2023/12/06 21:12
@Author :  Yufan Liu
@Desc   :  Generate features and path
'''
import os
from preprocessing.computeSurface import protonate, extractPDB, generate_surface 
from preprocessing.computeCharge import generate_charge
from preprocessing.computeHydro import generate_hydrophabicity
from preprocessing.computeAPBS import generate_apbs
from preprocessing.computeSI import generate_shapeindex
from preprocessing.computeDDC import generate_ddc

from preprocessing.geomesh import GeoMesh

from model_cache.polar_coords import generate_polar_coords

from typing import List
from tqdm import tqdm
import time
# for data preprocessing
# PDB -> precompute features and shortest path
# todo surface use pymol; shortest path use 
# load functions from preprocessing


class DataPrepare(object):
    def __init__(self, args, 
                 data_list: List,
                 collapse_rate=0.1,  # for mesh optimize collapse
                 ):
        self.args = args
        self.data_list = data_list
        self.collapse_rate = collapse_rate

    def run_protonate(self, args, data):
        input_file = os.path.join(args.raw_path, data.split('_')[0] + '.pdb')
        output_file = os.path.join(args.processed_path, data, 'raw.pdb') 
        protonate(args, input_file, output_file)

    def run_extractPDB(self, args, data, chain1, chain2):
        input_file = os.path.join(args.processed_path, data, 'raw.pdb')
        out_chain1 = os.path.join(args.processed_path, data, 'p1.pdb')
        out_chain2 = os.path.join(args.processed_path, data, 'p2.pdb')
        extractPDB(input_file, out_chain1, chain1)
        extractPDB(input_file, out_chain2, chain2)

    def run_surface(self, args, data, num):
        input_file = os.path.join(args.processed_path, data, f'p{num}.pdb')
        output_file = os.path.join(args.processed_path, data, f'p{num}.xyzrn')
        return generate_surface(args, input_file, output_file, cache=False)

    def run_charge(self, args, data, vertices, names, num):
        input_file = os.path.join(args.processed_path, data, f'p{num}.pdb')
        return generate_charge(input_file, vertices, names)

    def run_hydro(self, args, data, names, num):
        input_file = os.path.join(args.processed_path, data, f'p{num}.pdb')
        return generate_hydrophabicity(input_file, names)

    def run_apbs(self, args, data, vertex, num):
        input_file = os.path.join(args.processed_path, data, f'p{num}.pdb')
        return generate_apbs(args, input_file, vertex)


    def __call__(self):
        args = self.args
        data_list = self.data_list
            # "main" function in data prepare file for all in one process
        raw_path = args.raw_path
        processed_path = args.processed_path
        if not os.path.exists(processed_path):
            os.mkdir(processed_path) 
        
        print("Start processing data...")
        for data in data_list:
            start_time = time.time()
            # todo consider multi-processing in the future
            # data is like 1A0G_A_B
            pdb, chain1, chain2 = data.split('_')
            if not os.path.exists(os.path.join(processed_path, data)):
                os.mkdir(os.path.join(processed_path, data))

            # Stage I. generate surface 
            # 1. protonate
            self.run_protonate(args, data)
            print(f'running time for protonating: {time.time() - start_time}')

            # 2. extract PDB, all process after then should be executed in processed path
            self.run_extractPDB(args, data, chain1, chain2)
            print(f'running time for extracting PDB: {time.time() - start_time}')

            
            # process 2 chains separately from then on
            for num in range(2):
                num += 1
                # 3. generate surface, including non-standard amino acids
                vertices, faces, normals, names, areas = self.run_surface(args, data, num)
                print(f'running time for generating surface of component{num}: {time.time() - start_time}')

                # Stage II. generatue features
                # 1. generate electrons +/- donor and acceptor
                charge = self.run_charge(args, data, vertices, names, num)
                print(f'running time for computing charge of component{num}: {time.time() - start_time}')

                # 2. generate logp
                logp = self.run_hydro(args, data, names, num)
                print(f'running time for computing hydrophabicity of component{num}: {time.time() - start_time}')
                # print(len(logp) == len(vertices))
                # print(len(charge) == len(vertices))
                mesh = GeoMesh(vertex_matrix=vertices, face_matrix=faces, charge=charge, logp=logp)
                face_number = mesh.current_mesh().face_number()
                mesh.meshing_decimation_quadric_edge_collapse(targetfacenum=int(self.collapse_rate * face_number))
                mesh.meshing_repair_non_manifold_edges()  # remove edges
                #mesh.meshing_remove_duplicate_faces()
                #mesh.meshing.remove_duplicate_vertices()
                mesh.meshing_remove_unreferenced_vertices() # done fix by this
                mesh.apply_coord_taubin_smoothing()

                mesh.update_feature()

                # 3. generate APBS feature, chemical features ended here # note: use fine-mesh
                apbs = self.run_apbs(args, data, mesh.current_mesh().vertex_matrix(), num)
                print(f'running time for computing APBS charge of component{num}: {time.time() - start_time}')
                mesh.set_attribute('apbs', apbs)
                
                # initial a mesh, add 3 features above
                # Polar coords 
                # 4. shape index # todo
                rho, theta, neighbor_id, neighbor_mask = generate_polar_coords(mesh)
                mesh.set_attribute('rho', rho)
                mesh.set_attribute('theta', theta)
                mesh.set_attribute('neighbor_id', neighbor_id)
                mesh.set_attribute('neighbor_mask', neighbor_mask)
                
                mesh = generate_shapeindex(mesh)
                print(f"running time for generating polar coords & shape index of component {num}: {time.time() - start_time}")
                
                # 5. distance dependent curvature
                mesh = generate_ddc(mesh)
                print(f"running time for generating DDC of component {num}: {time.time() - start_time}")




            
