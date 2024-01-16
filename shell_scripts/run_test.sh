#!/bin/bash
#SBATCH --cpus-per-gpu 20
#SBATCH --gpus-per-node a100:1
#SBATCH --mem 24G
#SBATCH --time 00:10:00
#SBATCH --output=train_tmp_log/%J_testing.out
#SBATCH --error=train_tmp_log/%J_testing.err
#SBATCH --mail-type=ALL

/home/liuy0n/miniforge3/envs/pth/bin/python utilities/run_test.py \
--experiment_name test \
--dataset_path data/benchmark_chemical_dist/chemicalnet_dataset \
