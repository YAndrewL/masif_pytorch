#!/bin/bash
#SBATCH --cpus-per-gpu 20
#SBATCH --gpus-per-node a100:1
#SBATCH --mem 24G
#SBATCH --time 12:00:00
#SBATCH --output=train_tmp_log/%J_training.out
#SBATCH --error=train_tmp_log/%J_training.err
#SBATCH --mail-type=ALL

/home/liuy0n/miniforge3/envs/pth/bin/python main.py --experiment_name mask_chem --dataset_path benchmark_dataset --feature_mask 1 1 0 0 0