import Bio
from Bio.PDB import * 
import sys
import importlib
import os

def down_load_pdb(root, pdb_id):
    # download PDB file from server， leave for now, we have local files
    data_root = root

    if not os.path.exists(masif_opts['raw_pdb_dir']):
        os.makedirs(masif_opts['raw_pdb_dir'])

    if not os.path.exists(masif_opts['tmp_dir']):
        os.mkdir(masif_opts['tmp_dir'])

    in_fields = sys.argv[1].split('_')
    pdb_id = in_fields[0]

    # Download pdb 
    download = sys.argv[2]
    if download == 'Yes':
        pdbl = PDBList()
        pdb_filename = pdbl.retrieve_pdb_file(pdb_id, pdir=masif_opts['tmp_dir'])
    elif download == 'No':
        parser = PDBParser()
        pdb_filename = masif_opts['raw_pdb_dir']+"/"+pdb_id+".pdb"
