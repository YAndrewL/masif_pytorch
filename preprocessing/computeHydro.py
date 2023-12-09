# -*- coding: utf-8 -*-
'''
@File   :  computeHydro.py
@Time   :  2023/12/09 10:27
@Author :  Yufan Liu
@Desc   :  Hydrophobility (here use logp) feature
'''
# todo THIS used too many loops, but I'm lazy to modify this now, and compute them in advance, idc
import Bio.PDB as biopdb
import numpy as np


def generate_hydro(infilename):
    # split PDB file
    hydro = 
    parser = biopdb.PDBParser(QUIET=True)
    strtucture = parser.get_structure()
    residues = biopdb.Selection.unfold_entities()
    for residue in residues:


parser = biopdb.PDBParser(QUIET=True)
struct = parser.get_structure('test', 'chainI.pdb')
model = biopdb.Selection.unfold_entities(struct, 'R')
print(model.het)
# Select residues to extract and build new structure
structBuild = biopdb.StructureBuilder.StructureBuilder()
structBuild.init_structure("output")