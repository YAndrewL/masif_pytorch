# -*- coding: utf-8 -*-
'''
@File   :  arguments.py
@Time   :  2023/12/06 21:24
@Author :  Yufan Liu
@Desc   :  globe arguments
'''

import argparse
parser = argparse.ArgumentParser()

# data set paths
parser.add_argument("--data_path", type=str, default="data/")
parser.add_argument('--raw_path', type=str, default="data/raw_pdbs")
parser.add_argument('--processed_path', type=str, default="data/processed")
parser.add_argument('--dataset_path', type=str, default="dataset/")


# sotfware
parser.add_argument("--APBS_BIN", type=str, 
                    default="/home/liuy0n/tools/APBS-3.4.1.Linux/bin/apbs")
parser.add_argument("--MULTIVALUE_BIN", type=str, 
                    default="/home/liuy0n/tools/APBS-3.4.1.Linux/share/apbs/tools/bin/multivalue")
parser.add_argument("--PDB2PQR_BIN", type=str, 
                    default="/home/liuy0n/tools/pdb2pqr-linux-bin64-2.1.1/pdb2pqr")
parser.add_argument("--REDUCE_BIN", type=str, 
                    default="/home/liuy0n/tools/reduce/build/bin/reduce")  # prebuild by yourself
# parser.add_argument(REDUCE_HET_DICT=/home/liuy0n/tools/reduce/build/bin/reduce_wwPDB_het_dict.txt export to shell
parser.add_argument("--MSMS_BIN", type=str, 
                    default="/home/liuy0n/tools/msms/msms.x86_64Linux2.2.6.1")
parser.add_argument("--PDB2XYZRN", type=str, 
                    default="/home/liuy0n/tools/msms/pdb_to_xyzrn")  # not used 
# parser.add_argument(LD_LIBRARY_PATH=/home/liuy0n/tools/APBS-3.4.1.Linux/lib/  # export to shell


# parameters
parser.add_argument("--random_seed", 
                    type=int, 
                    default=42, 
                    help="seed.") 
parser.add_argument("--training_split", 
                    type=float, 
                    default=0.9
                    ) 
parser.add_argument("--max_vertex", 
                    type=int, 
                    default=200, 
                    help="Max vertex neighbor for input feature.") 


# for shape complementarity
parser.add_argument("--sc_w", 
                    type=int, 
                    default=0.25, 
                    help="")  
parser.add_argument("--sc_interaction_cutoff", 
                    type=float, 
                    default=1.5, 
                    help="Less than this value will be considered as interaction")
parser.add_argument("--sc_radius", 
                    type=int, 
                    default=12, 
                    help="patch radius")
parser.add_argument("--max_sc_filt",
                    type=float,
                    default=1.0,
                    help="Max sc value for positive filter")
parser.add_argument("--min_sc_filt",
                    type=float,
                    default=0.5,
                    help="Min sc value for positive filter")
parser.add_argument("--pos_interface_cutoff",
                    type=float,
                    default=1.0,
                    help="Distance cutoff for pos/neg define")

# model and dataset parameters for trainer.
parser.add_argument("--model_path",
                    type=str,
                    default="model",
                    help="default path for model logger and save")


parser.add_argument("--data_list",
                    type=str,
                    help="tmp used for data prepare")
parser.add_argument("--batch_size",
                    type=int,
                    default=64,
                    help="Batch size for training")
parser.add_argument("--device",
                    type=str,
                    default='cuda',
                    help="device for model")
parser.add_argument("--max_distance", 
                    type=float, 
                    default=12.0, 
                    help="Max patch radius for searching") 
parser.add_argument("--learning_rate",
                    type=float,
                    default=1e-3)
parser.add_argument("--epochs",
                    type=int,
                    default=10000)
parser.add_argument("--test_epochs",
                    type=int,
                    default=100)

parser.add_argument("--pos_thresh",
                    type=float,
                    default=0.0,
                    help="Threshold for positive samples")
parser.add_argument("--neg_thresh",
                    type=float,
                    default=10.0,
                    help="Threshold for negative samples")

# initialization model
parser.add_argument("--n_thetas",
                    type=int,
                    default=16,
                    help="Number of thetas for grid generation")
parser.add_argument("--n_rhos",
                    type=int,
                    default=5,
                    help="Number of rhos for grid generation")                                     
parser.add_argument("--n_rotations",
                    type=int,
                    default=16,
                    help="Number of rotations for feature max-out")   
parser.add_argument("--n_features",
                    type=int,
                    default=5,
                    help="Number of features")   
