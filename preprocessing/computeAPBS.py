# -*- coding: utf-8 -*-
'''
@File   :  computeAPBS.py
@Time   :  2023/12/10 12:05
@Author :  Yufan Liu
@Desc   :  Compute 
'''


import os
import numpy
from subprocess import Popen, PIPE
import time

"""
computeAPBS.py: Wrapper function to compute the Poisson Boltzmann electrostatics for a surface using APBS.
Pablo Gainza - LPDI STI EPFL 2019
This file is part of MaSIF.
Released under an Apache License 2.0
"""

def generate_apbs(args, infilename, vertices):
    """
        Calls APBS, pdb2pqr, and multivalue and returns the charges per vertex
    """
    abs_root = os.getcwd()  # APBS related calling need this
    infilename = os.path.join(abs_root, infilename)

    tmp_file_base = ''.join(infilename.split('.')[:-1])  # /path/path/p_num


    directory = '/'.join(tmp_file_base.split('/')[:-1])  # /path/path/


    pqr_args = [
        args.PDB2PQR_BIN,
        "--ff=parse",
        "--whitespace",
        "--noopt",
        "--apbs-input",
        infilename,
        tmp_file_base + '.pqr',
    ]  # todo PDB2PQR will ignore incomplete residues
    p2 = Popen(pqr_args, stdout=PIPE, stderr=PIPE, cwd=directory)
    stdout, stderr = p2.communicate()
    # should excuted in the same path
    apbs_args = [args.APBS_BIN, tmp_file_base + ".in"]
    p2 = Popen(apbs_args, stdout=PIPE, stderr=PIPE, cwd=directory)
    stdout, stderr = p2.communicate()

    vertfile = open(tmp_file_base + ".csv", "w")
    for vert in vertices:
        vertfile.write("{},{},{}\n".format(vert[0], vert[1], vert[2]))
    vertfile.close()

    multivalue_args = [
        args.MULTIVALUE_BIN,
        tmp_file_base + ".csv",
        tmp_file_base + ".pqr.dx",
        tmp_file_base + "_out.csv",
    ]
    p2 = Popen(multivalue_args, stdout=PIPE, stderr=PIPE, cwd=directory)
    stdout, stderr = p2.communicate()
    # Read the charge file
    chargefile = open(tmp_file_base + "_out.csv")
    charges = numpy.array([0.0] * len(vertices))
    for ix, line in enumerate(chargefile.readlines()):
        charges[ix] = float(line.split(",")[3])

    remove_fn = tmp_file_base
    os.remove(remove_fn + '.csv')
    os.remove(remove_fn + '.pqr.dx')
    os.remove(remove_fn + '.in')
    os.remove(remove_fn + '-input.p')
    os.remove(remove_fn + '_out.csv')
    os.remove(remove_fn + '.pqr')
    os.remove(directory + '/io.mc')
    return charges

