#!/bin/bash
#SBATCH --cpus-per-gpu 20
#SBATCH --gpus-per-node a100:1
#SBATCH --mem 24G
#SBATCH --time 24:00:00
#SBATCH --output=logs/%J_training.out
#SBATCH --error=logs/%J_training.err
#SBATCH --mail-type=ALL

/home/liuy0n/miniforge3/envs/pth/bin/python main.py \
--experiment_name benchmark_ChemicalNet_dist --epochs 200 \
--dataset_path data/benchmark_chemical_dist/chemicalnet_dataset \
--chemical_net siamese

/home/liuy0n/miniforge3/envs/pth/bin/python main.py \
--experiment_name peptide_tune --epochs 50 --test_epochs 1 \
--dataset_path data/cyclic_nonstd_peptide/peptide_dataset \ 
--mode train --cache_model /ibex/user/liuy0n/codes/masif_pth/experiments/benchmark_ChemicalNet_dist/01-14-11-12/model_large.pth