#!/bin/bash
#SBATCH --nodes 1
# #SBATCH --partition=serial
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 2G
#SBATCH --time 00:30:00
#SBATCH --array=1-100
#SBATCH --output=data_prepare_logs/_download_pdb.%A_%a.out
#SBATCH --error=data_prepare_logs/_download_pdb.%A_%a.err
#SBATCH --mail-type=ALL


START=$(( (SLURM_ARRAY_TASK_ID - 1) * 298 + 1 ))
END=$(( SLURM_ARRAY_TASK_ID * 298 ))

if [ $SLURM_ARRAY_TASK_ID -eq 100 ]; then
    END=29769
fi


i=1
while read p; do
    if [ $i -ge $START ] && [ $i -le $END ]; then
        /home/liuy0n/miniforge3/envs/pth/bin/python split.py $p
    fi
    i=$((i+1))
done < peptide_data/list/reverse_full.txt

 