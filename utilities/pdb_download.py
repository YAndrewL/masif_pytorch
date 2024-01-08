# -*- coding: utf-8 -*-
'''
@File   :  pdb_download.py
@Time   :  2024/01/08 13:56
@Author :  Yufan Liu
@Desc   :  Download PDB file.
'''

import os
import sys
import Bio.PDB as biopdb

pdb = sys.argv[1]
print(pdb)
if os.path.exists(os.path.join("./peptide_data/raw_pdbs/", pdb)):
    exit()


pdb = pdb.split ('.')[0]
dir = './peptide_data/raw_pdbs/'
pdbl = biopdb.PDBList()


pdbl.retrieve_pdb_file(pdb, pdir=dir, file_format='pdb')
    # rename to pdb file
downloaded = os.path.join(dir, 'pdb' + pdb.lower() + '.ent')
os.rename(downloaded, os.path.join(dir, pdb.upper() + '.pdb'))



