#!/bin/bash
#SBATCH --cpus-per-gpu 20
#SBATCH --gpus-per-node a100:1
#SBATCH --mem 24G
#SBATCH --time 24:00:00
#SBATCH --output=logs/%J_training.out
#SBATCH --error=logs/%J_training.err
#SBATCH --mail-type=ALL

# /home/liuy0n/miniforge3/envs/pth/bin/python main.py \
# --experiment_name benchmark_ChemicalNet_dist --epochs 150 \
# --dataset_path data/benchmark_chemical_dist/chemicalnet_dataset \
# --chemical_net siamese \
# --vocab_length 6

# fine tune peptide 
/home/liuy0n/miniforge3/envs/pth/bin/python main.py \
--experiment_name peptide_tune --epochs 50 --test_epochs 1 \
--dataset_path data/cyclic_nonstd_peptide/peptide_dataset \
--mode train --cache_model /ibex/user/liuy0n/codes/masif_pth/experiments/benchmark_ChemicalNet_dist/01-14-11-12/model_large.pth

# test the model
# python main.py --experiment_name test \
# --dataset_path data/benchmark_chemical_dist/chemicalnet_dataset \
# --chemical_net flip --cache_model experiments/benchmark_ChemicalNet_dist/01-16-11-41/model.pth --mode test

# make inference
python inference.py --experiment_name infer \
--chemical_net flip --cache_model experiments/peptide_tune/01-16-23-00/model.pth \
--inf_target_path pep_library_one/target_7yv1.npy --inf_binder_path pep_library_one/pep_feat_7to15.npy \
--inf_save_path analysis/KRAS
