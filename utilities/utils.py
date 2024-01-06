# -*- coding: utf-8 -*-
'''
@File   :  utils.py
@Time   :  2023/12/23 19:19
@Author :  Yufan Liu
@Desc   :  Some helper functions for experiments
'''

import os
import shutil

def add_chain_id_to_pdb(file_path, save_path, chain_id):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(save_path, 'w') as file:
        for line in lines:
            if line.startswith("ATOM"):
                line = line[:21] + chain_id + line[22:]
                file.write(line)
            elif line.startswith("CRY"):
                continue
            else:
                file.write(line)


def merge_pdb_file(file_a, file_b, output_file):
    with open(file_a, 'r') as fa, open(file_b, 'r') as fb, open(output_file, 'w') as outfile:
        for line in fa.readlines():
            if 'END' not in line:
                outfile.write(line)
        blines = fb.readlines()
        for line in blines:
            if 'ATOM' in line:
                outfile.write(line)
        outfile.write("TER\n")
        for line in blines:
            if 'CONECT' in line:
                outfile.write(line)
        outfile.write("END\n")

def remove_incompelete_folders(processed_path):
    check = sorted(
        ['preprocess.log',
         'p1_input_feat.npy',
         'p2_input_feat.npy',
         'p2.ply',
         'p1.pdb',
         'p2.pdb',
         'p1.ply',
         'raw.pdb']
        )
    for data in os.listdir(processed_path):
        if sorted(os.listdir(os.path.join(processed_path, data))) != check:
            shutil.rmtree(os.path.join(processed_path, data))
            print(f"Delete incomplete folder {data}, due to APBS computing issue.")