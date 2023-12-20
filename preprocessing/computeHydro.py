# -*- coding: utf-8 -*-
'''
@File   :  computeHydro.py
@Time   :  2023/12/09 10:27
@Author :  Yufan Liu
@Desc   :  Hydrophobility (here use logp) feature
'''
import Bio.PDB as biopdb
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem


def generate_hydrophabicity(args, infilename, names):
    # split PDB file
    #hydro = np.zeros(len(names))  # just a length, no matter who is passed
    parser = biopdb.PDBParser(QUIET=True)
    strtucture = parser.get_structure(infilename, infilename)
    residues = biopdb.Selection.unfold_entities(strtucture, 'R')
    hydro = dict()
    for residue in residues:
        """
        # res_id = residue.get_id()
        # if res_id[0] != ' ':
        #     res_id = (' ', res_id[1], res_id[2])
        # io = biopdb.PDBIO()
        # io.set_structure(residue)
        # tmp = ''.join(infilename.split('.')[:-1]) + '.tmp.' + str(res_id)
        # io.save(tmp)
        # mol = Chem.rdmolfiles.MolFromPDBFile(tmp)
        # logp_mol = AllChem.CalcCrippenDescriptors(mol)[0]  
        # # here the first is logp, then normalize to -1 ~ 1
        # os.remove(tmp)
        """
        # code above are compute by structure, move to by sdf of amino libaray
        res_id = residue.get_id()
        if res_id[0] != ' ':
            res_id = (' ', res_id[1], res_id[2])
        residue_name = residue.get_resname()
        mol = Chem.SDMolSupplier(os.path.join(args.residue_lib, f"{residue_name}_ideal.sdf"))[0]
        if mol is None:
            raise RuntimeError(f"In valid SDF file of residue {residue_name} in {infilename}, location {res_id}")
        logp_mol = AllChem.CalcCrippenDescriptors(mol)[0]
        hydro[res_id] = logp_mol
    # assign to each vertex
    logp = np.zeros(len(names))
    for idx, name in enumerate(names):
        fields = name.split('_')
        if fields[2] == "x":
            fields[2] = " "
        match = (" ", int(fields[1]), fields[2])
        logp[idx] = hydro[match]
    return logp
