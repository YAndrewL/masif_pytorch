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
parser.add_argument("--data_root", type=str, default="data/")
parser.add_argument('--raw_path', type=str, default="data/raw_pdbs")
parser.add_argument('--processed_path', type=str, default="data/processed")


# sotfware
parser.add_argument("--APBS_BIN", type=str, default="/home/liuy0n/tools/APBS-3.4.1.Linux/bin/apbs")
parser.add_argument("--MULTIVALUE_BIN", type=str, default="/home/liuy0n/tools/APBS-3.4.1.Linux/share/apbs/tools/bin/multivalue")
parser.add_argument("--PDB2PQR_BIN", type=str, default="/home/liuy0n/tools/pdb2pqr-linux-bin64-2.1.1/pdb2pqr")
#parser.add_argument(PATH=$PATH:/home/liuy0n/tools/reduce/build/bin/
# parser.add_argument(REDUCE_HET_DICT=/home/liuy0n/tools/reduce/build/bin/reduce_wwPDB_het_dict.txt export to shell
parser.add_argument("--MSMS_BIN", type=str, default="/home/liuy0n/tools/msms/msms.x86_64Linux2.2.6.1")
parser.add_argument("--PDB2XYZRN", type=str, default="/home/liuy0n/tools/msms/pdb_to_xyzrn")
# parser.add_argument(LD_LIBRARY_PATH=/home/liuy0n/tools/APBS-3.4.1.Linux/lib/  # export to shell

