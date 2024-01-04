# -*- coding: utf-8 -*-
'''
@File   :  screen.py
@Time   :  2024/01/03 19:14
@Author :  Yufan Liu
@Desc   :  Screen target pocket with a prepared library
'''

# target KRAS, with peptide library


# test
from arguments import parser
from data_prepare import process_single
import sys

args = parser.parse_args()

# following part is process single, use separately.
# pdb = args.data_list
# process_single(args, pdb, 'analysis/KRAS/raw_files', 'analysis/KRAS/raw_feature')

# target: 7YV1_A, library: peptide

