# -*- coding: utf-8 -*-
'''
@File   :  utils.py
@Time   :  2023/12/23 19:19
@Author :  Yufan Liu
@Desc   :  Some helper functions for experiments
'''


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
    import os
    import shutil
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

def write_to_pointcloud(coordinates, bfactors, output):
    """
    Write coordinates into a point clould PDB file, using biopython
    coordinates: ndarray of shape [N, 3]
    bfactors: ndarray of shape [N, ]
    """
    from Bio.PDB import PDBIO, Structure, Model, Chain, Residue, Atom
    
    assert len(coordinates) == len(bfactors), "File length not match."
    structure = Structure.Structure("1")
    model = Model.Model(0)
    structure.add(model)
    chain = Chain.Chain("A")
    model.add(chain)

    for i, bf in enumerate(bfactors):
        res = Residue.Residue((' ', i+1, ' '), 'XYZ', '')
        chain.add(res)
        
        atom = Atom.Atom('H', coordinates[i], 1.0, bf, ' ', 'H', i+1, 'H')
        res.add(atom)    

    io = PDBIO()
    io.set_structure(structure)
    io.save(output)

def find_patch(target, cloud, cutoff=12):
    """
    Find all vertices in a patch given center
    target: shape of [1, 3]
    cloud: shape of [M, 3]
    """
    from scipy.spatial import distance
    import numpy as np

    assert len(target.shape) == 2, "Dim not matched, expand dim if only 1 target is passed."
    for i in range(len(target)):
        center = target[i]
        dist = distance.cdist(cloud, target)  # [M, N]
        idx = np.where(dist < cutoff)[0]
    return idx  # [Number of vertices in patch with cutoff]

def color_ply(infile, outfile, color_id, mode='single'):
    """
    Add color to selected 
    """





