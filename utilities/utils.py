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

def is_number(s):
    try:
        float(s)
        return True
    except:
        return False


def color_ply(infile, outfile, color_id, mode='single', delete=True):
    """
    Add color to selected indices
    color_id should be a list or ndarray in int.
    """
    # todo set mode to a gradient color scale
    import pymeshlab
    import numpy as np

    mesh = pymeshlab.MeshSet()
    mesh.load_new_mesh(infile)
    
    vertex = mesh.current_mesh().vertex_matrix()
    color = np.full_like(vertex, 255)  # all in white
    #print(color.shape)
    for i in range(len(vertex)):
        if i in color_id:
            color[i, :] = [255, 0, 0]
    
    # manually write in ply, only support pure xyz ply file.
    counter = 0
    with open(outfile, 'w') as f:
        for line in open(infile).readlines():
            if line.startswith("element vertex"):
                vertex_number = line.strip().split(" ")[-1]
            if line.startswith("element face"):
                face_number = line.strip().split(" ")[-1]            
                break
        
        # write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment VCGLIB generated\n")
        f.write(f"element vertex {vertex_number}\n")
        f.write("property double x\n")
        f.write("property double y\n")
        f.write("property double z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {face_number}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for i, line in enumerate(open(infile).readlines()):
            line = line.strip()
            head_str = line.split(" ")[0]
            if not is_number(head_str):
                continue
            if is_number(head_str) and counter <= len(color) - 1:
                f.write(line + ' ')
                fillin = color[counter]
                f.write(str(int(fillin[0])) + ' ')
                f.write(str(int(fillin[1])) + ' ')
                f.write(str(int(fillin[2])) + '\n')
                counter += 1
            else:
                f.write(line + '\n')

def expand_vocab_size(model_path, new_size=8):
    """
    increase the vocabulary size
    """
    import torch
    import torch.nn as nn
    
    stat_dict = torch.load(model_path)
    embedding = stat_dict['atomtype_embedding'].weight.data
    old_size = embedding.shape[0]
    init_weight = nn.Embedding(new_size, 1).weight.data
    init_weight[:old_size, :] = embedding
    stat_dict['atomtype_embedding'] = init_weight
    
    return stat_dict


def generate_transform_matrix():
    import numpy as np
    theta = np.radians(np.random.uniform(0, 360))
    phi = np.radians(np.random.uniform(0, 360))
    psi = np.radians(np.random.uniform(0, 360))

    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                   [np.sin(theta), np.cos(theta), 0],
                   [0, 0, 1]])

    Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
                   [0, 1, 0],
                   [-np.sin(phi), 0, np.cos(phi)]])

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(psi), -np.sin(psi)],
                   [0, np.sin(psi), np.cos(psi)]])
    
    return np.dot(Rz, np.dot(Ry, Rx)), np.random.rand(3) * 2  # scale to 2 Angstrom

def transform_pdb(pdb_file, out_file, chain=None, rotate_translate=None):
    """
    This is typically for benchmark
    random rotate and translate the binder
    to test whether algorithm can recover the conformation
    """
    import Bio.PDB as biopdb
    
    from Bio.PDB import Atom
    
    parser = biopdb.PDBParser()
    structure = parser.get_structure("pdb", pdb_file)
    if rotate_translate:
        rotate, translate = rotate_translate
    else:
        rotate, translate = generate_transform_matrix()
    
    if chain:
        chain_to_transform = structure[0][chain]
        for residue in chain_to_transform:
            for atom in residue:
                atom.transform(rotate, translate)
    else:
        for atom in structure.get_atoms():
            atom.transform(rotate, translate)

    
    io = biopdb.PDBIO()
    io.set_structure(structure)
    io.save(out_file)
    
def compute_RMSD(a, b):
    """
    RMSD of two ligands
    """
    import numpy as np
    rmsd = np.sqrt(
        np.mean(
            np.square(
                np.linalg.norm(
                    a - b,
                    axis=1,
                )
            )
        )
    )
    return rmsd

def screen_lib(target, binder_candidate, names, out_file, topN=None, target_site=None):
    """
    Use descriptor to generate a screen table
    target, binder_candidate and names are all file paths
    awa 
    """
    import numpy as np
    import pickle
    from scipy.spatial import distance
    
    if target_site:
        target_desc = np.load(target)[target_site: target_site + 1, :]  # selete the target patch
    else:
        target_desc = np.load(target)
        
    binder_desc = np.load(binder_candidate)
    pep_names = pickle.load(open(names, "rb"))
    
    all_distance = distance.cdist(target_desc, binder_desc)  # [M, N] 
    all_distance[np.isnan(all_distance)] = 99  # just in case
    
    target_idx, binder_idx = np.unravel_index(np.argsort(all_distance, axis=None), shape=all_distance.shape)
    
    with open(out_file, "w") as f:
        f.write("Pep_ID\t")
        f.write("Target\t")
        f.write("Binder\t")
        f.write("Dist\n")
        
        
        if topN:
            counter = 0
        for i in range(len(target_idx)):
            f.write(pep_names[binder_idx[i]] + '\t')
            f.write(str(target_idx[i]) + '\t')
            f.write(str(binder_idx[i]) + '\t')
            f.write(str(round(all_distance[target_idx[i]][binder_idx[i]], 6)) + '\n')
            
            if topN:
                counter += 1
                if counter >= topN:
                    break