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
parser.add_argument("--APBS_BIN", type=str, default="/home/liuy0n/tools/APBS-3.4.1.Linux/bin/apbs")
parser.add_argument("--MULTIVALUE_BIN", type=str, default="/home/liuy0n/tools/APBS-3.4.1.Linux/share/apbs/tools/bin/multivalue")
parser.add_argument("--PDB2PQR_BIN", type=str, default="/home/liuy0n/tools/pdb2pqr-linux-bin64-2.1.1/pdb2pqr")
parser.add_argument("--REDUCE_BIN", type=str, default="/home/liuy0n/tools/reduce/build/bin/reduce")  # prebuild by yourself
# parser.add_argument(REDUCE_HET_DICT=/home/liuy0n/tools/reduce/build/bin/reduce_wwPDB_het_dict.txt export to shell
parser.add_argument("--MSMS_BIN", type=str, default="/home/liuy0n/tools/msms/msms.x86_64Linux2.2.6.1")
parser.add_argument("--PDB2XYZRN", type=str, default="/home/liuy0n/tools/msms/pdb_to_xyzrn")  # not used 
# parser.add_argument(LD_LIBRARY_PATH=/home/liuy0n/tools/APBS-3.4.1.Linux/lib/  # export to shell


# parameters
parser.add_argument("--random_seed", 
                    type=int, 
                    default=42, 
                    help="seed.") 
parser.add_argument("--max_vertex", 
                    type=int, 
                    default=200, 
                    help="Max vertex neighor for input feature.") 
# for shape complementarity
parser.add_argument("--sc_w", 
                    type=int, 
                    default=0.25, 
                    help="")  # todo what's for? 
parser.add_argument("--sc_interaction_cutoff", 
                    type=float, 
                    default=1.5, 
                    help="Less than this value will be considered as interaction")
parser.add_argument("--sc_radius", 
                    type=int, 
                    default=12, 
                    help="patch radius")

# dataset parameters
