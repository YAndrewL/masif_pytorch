# -*- coding: utf-8 -*-
'''
@File   :  utils.py
@Time   :  2023/12/08 10:41
@Author :  Yufan Liu
@Desc   :  Compute from .pdb file to .vertex file by MSMS to generate surface
'''

import pymol
from pymol import cmd
import numpy as np
import os
from subprocess import Popen, PIPE


def generate_xyzr(infilename, outfilename, selection='all'):
    # pymol.finish_launching() # lyf I'm quite sure what will happen without this function
    # but works now
    cmd.load(infilename)
    with open(outfilename, 'w') as f:
        model = cmd.get_model(selection)
        for atom in model.atom:
            x, y, z = atom.coord
            radius = atom.vdw
            f.write(f"{x} {y} {z} {radius} 1 {atom.chain}_{atom.resi}_{atom.resn}_{atom.name} \n")
            # the last field is important to denote the atom and residue for feature computing
    
    # print(f"File '{outfilename}' has been written with {len(model.atom)} atoms.")
    return outfilename

def read_msms(file_root):
    # read the surface from the msms output. MSMS outputs two files: {file_root}.vert and {file_root}.face
    
    vertfile = open(file_root + ".vert")
    meshdata = (vertfile.read().rstrip()).split("\n")
    vertfile.close()

    # Read number of vertices.
    count = {}
    header = meshdata[2].split()
    count["vertices"] = int(header[0])
    ## Data Structures
    vertices = np.zeros((count["vertices"], 3))
    normalv = np.zeros((count["vertices"], 3))
    atom_id = [""] * count["vertices"]
    res_id = [""] * count["vertices"]
    for i in range(3, len(meshdata)):
        fields = meshdata[i].split()
        vi = i - 3
        vertices[vi][0] = float(fields[0])
        vertices[vi][1] = float(fields[1])
        vertices[vi][2] = float(fields[2])
        normalv[vi][0] = float(fields[3])
        normalv[vi][1] = float(fields[4])
        normalv[vi][2] = float(fields[5])
        atom_id[vi] = fields[7]
        res_id[vi] = fields[9]
        count["vertices"] -= 1

    # Read faces.
    facefile = open(file_root + ".face")
    meshdata = (facefile.read().rstrip()).split("\n")
    facefile.close()

    # Read number of vertices.
    header = meshdata[2].split()
    count["faces"] = int(header[0])
    faces = np.zeros((count["faces"], 3), dtype=int)
    normalf = np.zeros((count["faces"], 3))

    for i in range(3, len(meshdata)):
        fi = i - 3
        fields = meshdata[i].split()
        faces[fi][0] = int(fields[0]) - 1
        faces[fi][1] = int(fields[1]) - 1
        faces[fi][2] = int(fields[2]) - 1
        count["faces"] -= 1

    assert count["vertices"] == 0
    assert count["faces"] == 0

    return vertices, faces, normalv, res_id

