#!/bin/bash
#SBATCH --nodes 1
# #SBATCH --partition=serial
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 12G
#SBATCH --time 24:00:00
#SBATCH --array=1-100
#SBATCH --output=data_prepare_logs/_masif_precompute.%A_%a.out
#SBATCH --error=data_prepare_logs/_masif_precompute.%A_%a.err
#SBATCH --mail-type=ALL


START=$(( (SLURM_ARRAY_TASK_ID - 1) * 298 + 1 ))
END=$(( SLURM_ARRAY_TASK_ID * 298 ))

if [ $SLURM_ARRAY_TASK_ID -eq 100 ]; then
    END=29770
fi


export REDUCE_HET_DICT=/home/liuy0n/tools/reduce/build/bin/reduce_wwPDB_het_dict.txt
export LD_LIBRARY_PATH=/home/liuy0n/tools/APBS-3.4.1.Linux/lib/


i=1
while read p; do
    if [ $i -ge $START ] && [ $i -le $END ]; then
        /home/liuy0n/miniforge3/envs/pth/bin/python main.py --data_list $p --prepare_data True --dataset_path peptide_dataset --experiment_name prepare --data_path peptide_data --raw_path peptide_data/raw_pdbs --processed_path peptide_data/processed
    fi
    i=$((i+1))
done < peptide_data/list/reverse_full.txt

 