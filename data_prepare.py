# -*- coding: utf-8 -*-
'''
@File   :  data_prepare.py
@Time   :  2023/12/06 21:12
@Author :  Yufan Liu
@Desc   :  Generate features and path, features and Dataset class for training
'''
import os
from os.path import join as pjoin
import time
import random
from loguru import logger

from preprocessing.computeSurface import protonate, extractPDB, generate_surface 
from preprocessing.computeCharge import generate_charge
from preprocessing.computeHydro import generate_hydrophabicity
from preprocessing.computeAPBS import generate_apbs
from preprocessing.computeSI import generate_shapeindex
from preprocessing.computeDDC import generate_ddc
from preprocessing.computeSC import generate_shape_complementarity

from preprocessing.cacheCoord import generate_polar_coords
from preprocessing.cacheSample import generate_data_cache

from preprocessing.geomesh import GeoMesh

from dataset import SurfaceDataset, collate_fn
from torch.utils.data import DataLoader


class DataPrepare(object):
    def __init__(self, args, 
                 training_list=[],
                 testing_list=[],
                 data_list=None,
                 ):
        self.args = args
        self.training_list = training_list
        self.testing_list = testing_list
        self.data_list = training_list + testing_list

        self.collapse_rate = args.collapse_rate  # for mesh optimize collapse


        # some sanity check for dataset
        assert args.dataset_path.split("_")[-1] == "dataset", "Suffix should be dataset, check argument helper"
        if args.prepare_data:
            if data_list:
                self.data_list = data_list
            else:
                raise RuntimeError("data list cannot be empty under data peparation mode")


    def run_protonate(self, args, data):
        input_file = pjoin(args.raw_path, data.split('_')[0] + '.pdb')
        output_file = pjoin(args.processed_path, data, 'raw.pdb') 
        protonate(args, input_file, output_file)

    def run_extractPDB(self, args, data, chain1, chain2):
        input_file = pjoin(args.processed_path, data, 'raw.pdb')
        out_chain1 = pjoin(args.processed_path, data, 'p1.pdb')
        out_chain2 = pjoin(args.processed_path, data, 'p2.pdb')
        extractPDB(input_file, out_chain1, chain1)
        extractPDB(input_file, out_chain2, chain2)

    def run_surface(self, args, data, num):
        input_file = pjoin(args.processed_path, data, f'p{num}.pdb')
        output_file = pjoin(args.processed_path, data, f'p{num}.xyzrn')
        return generate_surface(args, input_file, output_file, cache=False)

    def run_charge(self, args, data, vertices, names, num):
        input_file = pjoin(args.processed_path, data, f'p{num}.pdb')
        return generate_charge(input_file, vertices, names)

    def run_hydro(self, args, data, names, num):
        input_file = pjoin(args.processed_path, data, f'p{num}.pdb')
        return generate_hydrophabicity(args, input_file, names)

    def run_apbs(self, args, data, vertex, num):
        input_file = pjoin(args.processed_path, data, f'p{num}.pdb')
        return generate_apbs(args, input_file, vertex)

    def preprocess(self):
        args = self.args
        data_list = self.data_list
            # "main" function in data prepare file for all in one process
        processed_path = args.processed_path
        if not os.path.exists(processed_path):
            os.mkdir(processed_path) 
        
        for data in data_list:
            start_time = time.time()
            # data is like 1A0G_A_B
            pdb, chain1, chain2 = data.split('_')
            if not os.path.exists(pjoin(processed_path, data)):
                os.mkdir(pjoin(processed_path, data))

            logfile = pjoin(processed_path, data, 'preprocess.log')
            handler = logger.add(logfile)
            logger.info(f"Start processing data {data}")

            # Stage I. generate surface 
            # 1. protonate
            self.run_protonate(args, data)
            logger.info(f'protonating take: {time.time() - start_time}')

            # 2. extract PDB, all process after then should be executed in processed path
            self.run_extractPDB(args, data, chain1, chain2)
            logger.info(f'extracting PDB take: {time.time() - start_time}')

            
            # process 2 chains separately from then on
            meshes = []
            for num in range(2):
                num += 1
                # 3. generate surface, including non-standard amino acids
                vertices, faces, normals, names, areas = self.run_surface(args, data, num)
                logger.info(f'generating surface of chain{num} take: {time.time() - start_time}')

                # Stage II. generatue features
                # 1. generate electrons +/- donor and acceptor
                charge = self.run_charge(args, data, vertices, names, num)
                logger.info(f'computing charge of chain{num} take: {time.time() - start_time}')

                # 2. generate logp
                logp = self.run_hydro(args, data, names, num)
                logger.info(f'computing hydrophabicity of chain{num} take: {time.time() - start_time}')
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
                logger.info(f'computing APBS charge of chain{num} take: {time.time() - start_time}')
                mesh.set_attribute('apbs', apbs)
                
                # initial a mesh, add 3 features above
                # Polar coords 
                # 4. shape index 
                rho, theta, neighbor_id, neighbor_mask = generate_polar_coords(args, mesh)
                mesh.set_attribute('rho', rho)
                mesh.set_attribute('theta', theta)
                mesh.set_attribute('neighbor_id', neighbor_id)
                mesh.set_attribute('neighbor_mask', neighbor_mask)
                
                mesh = generate_shapeindex(mesh)
                logger.info(f"generating polar coords & shape index of chain{num} take: {time.time() - start_time}")
                
                # 5. distance dependent curvature 
                mesh = generate_ddc(args, mesh)
                logger.info(f"generating DDC of chain{num} take: {time.time() - start_time}")

                mesh.normalize_features()
                #mesh.save_feature(pjoin(args.processed_path, data, f'p{num}_input_feat.npy'))
                #mesh.save_current_mesh(pjoin(args.processed_path, data, f'p{num}.ply'), binary=False, save_vertex_color=False)
                logger.info(f"Feature and polygon files saved for component{num}.")
                assert mesh.feat_norm_flag == True
                meshes.append(mesh)

            # ignore previous mesh
            # define positive, negatives
            mesh1, mesh2 = generate_shape_complementarity(args, meshes)  
            mesh1.save_feature(pjoin(args.processed_path, data, f'p1_input_feat.npy'))
            mesh2.save_feature(pjoin(args.processed_path, data, f'p2_input_feat.npy'))
            mesh1.save_current_mesh(pjoin(args.processed_path, data, f'p1.ply'), binary=False, save_vertex_color=False)
            mesh2.save_current_mesh(pjoin(args.processed_path, data, f'p2.ply'), binary=False, save_vertex_color=False)
            logger.success("feature preprocessing done, mesh saved.")
            logger.remove(handler)
        return True

    def cache(self):
        args = self.args
        train_path = pjoin(args.dataset_path, 'train')
        val_path = pjoin(args.dataset_path, 'val')
        test_path = pjoin(args.dataset_path, 'test')
        
        if not os.path.exists(train_path):
            os.mkdir(train_path)
        if not os.path.exists(val_path):
            os.mkdir(val_path)
        if not os.path.exists(test_path):
            os.mkdir(test_path)

        random.seed(args.random_seed)
        # leave 10% as val
        training_list = self.training_list
        random.shuffle(training_list)
        train_num = int(len(training_list) * self.args.training_split)
        
        trainset = training_list[:train_num]
        valset = training_list[train_num:]
        testset = self.testing_list

        start_time = time.time()
        for dataset, datapath in zip([trainset, valset, testset], [train_path, val_path, test_path]):
            generate_data_cache(args, pjoin(datapath), dataset)
        print(f"Data caching finished, time take: {time.time() - start_time}")
        return True
    
    def dataset(self, data_type:str, 
                batch_size:int, 
                pair_shuffle:bool) -> DataLoader:
        data = SurfaceDataset(args=self.args, 
                              dataset_type=data_type, 
                              pair_shuffle=pair_shuffle)
        if data_type == 'train':
            shuffle = True
        else: shuffle = False
        dset = DataLoader(data, batch_size=batch_size, 
                          collate_fn=collate_fn(), 
                          shuffle=shuffle,
                          num_workers=self.args.num_workers)
        return dset



    











            
