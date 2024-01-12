#!/bin/bash
#SBATCH --cpus-per-gpu 20
#SBATCH --gpus-per-node a100:1
#SBATCH --mem 24G
#SBATCH --time 24:00:00
#SBATCH --output=logs/%J_training.out
#SBATCH --error=logs/%J_training.err
#SBATCH --mail-type=ALL

/home/liuy0n/miniforge3/envs/pth/bin/python main.py \
--experiment_name benchmark_ChemicalNet --epochs 500 \
--dataset_path data/benchmark_chemical_net/chemicalnet_dataset \
--chemical_net True
