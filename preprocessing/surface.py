# -*- coding: utf-8 -*-
'''
@File   :  features.py
@Time   :  2023/12/07 13:44
@Author :  Yufan Liu
@Desc   :  to generte protein surface in .vertex file
'''

import pymol
from pymol import cmd
from preprocessing.computation.computeMSMS import generate_xyzr, read_msms
import time
import trimesh
import os 
from subprocess import Popen, PIPE
import Bio.PDB as biopdb
from Bio.SeqUtils import IUPACData
PROTEIN_LETTERS = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]

def protonate(infilename, outfilename):
    # protonate (i.e., add hydrogens) a pdb using reduce and save to an output file.
    # in_pdb_file: file to protonate.
    # out_pdb_file: output file where to save the protonated pdb file. 
    
    # Remove protons first, in case the structure is already protonated
    args = ["reduce", "-Trim", infilename]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(outfilename, "w")
    outfile.write(stdout.decode('utf-8').rstrip())
    outfile.close()
    # Now add them again.
    args = ["reduce", "-HIS", outfilename]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(outfilename, "w")
    outfile.write(stdout.decode('utf-8'))
    outfile.close()

def find_modified_amino_acids(path):
    """
    Contributed by github user jomimc - find modified amino acids in the PDB (e.g. MSE)
    """
    res_set = set()
    for line in open(path, 'r'):
        if line[:6] == 'SEQRES':
            for res in line.split()[4:]:
                res_set.add(res)
    for res in list(res_set):
        if res in PROTEIN_LETTERS:
            res_set.remove(res)
    return res_set
# Exclude disordered atoms.
class NotDisordered(biopdb.Select):
    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == "A"  or atom.get_altloc() == "1" 
    
def extractPDB(
    infilename, outfilename, chain_ids=None 
):
    # extract the chain_ids from infilename and save in outfilename. 
    parser = biopdb.PDBParser(QUIET=True)
    struct = parser.get_structure(infilename, infilename)
    model = biopdb.Selection.unfold_entities(struct, "M")[0]
    chains = biopdb.Selection.unfold_entities(struct, "C")
    # Select residues to extract and build new structure
    structBuild = biopdb.StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()

    # Load a list of non-standard amino acid names -- these are
    # typically listed under HETATM, so they would be typically
    # ignored by the orginal algorithm
    modified_amino_acids = find_modified_amino_acids(infilename)

    for chain in model:
        # print(f"chain id is {chain.get_id()}")
        if (
            chain_ids == None
            or chain.get_id() in chain_ids
        ):
            structBuild.init_chain(chain.get_id())
            for residue in chain:
                het = residue.get_id()
                if het[0] == " ":
                    outputStruct[0][chain.get_id()].add(residue)
                elif het[0][-3:] in modified_amino_acids:
                    outputStruct[0][chain.get_id()].add(residue)

    # Output the selected residues
    pdbio = biopdb.PDBIO()
    pdbio.set_structure(outputStruct)
    pdbio.save(outfilename, select=NotDisordered())


def generate_surface(args, infilename, outfilename, cache=False):
    # pymol to generate xyzr, and MSMS to generate vertex files
    generate_xyzr(infilename, outfilename)
    file_base = ''.join(infilename.split('.')[:-1])

    msms_arg = [args.MSMS_BIN, "-density", "3.0", "-hdensity", "3.0", "-probe",\
                    "1.5", "-if",outfilename,"-of",file_base, "-af", file_base]
    #print msms_bin+" "+`args`
    p2 = Popen(msms_arg, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()

    vertices, faces, normals, names = read_msms(file_base)
    areas = {}
    ses_file = open(file_base+".area")
    next(ses_file) # ignore header line
    for line in ses_file:
        fields = line.split()
        areas[fields[3]] = fields[1]

    # Remove temporary files. 
    if not cache:
        os.remove(file_base+'.area')
        os.remove(file_base+'.xyzrn')
        os.remove(file_base+'.vert')
        os.remove(file_base+'.face')
    return vertices, faces, normals, names, areas


