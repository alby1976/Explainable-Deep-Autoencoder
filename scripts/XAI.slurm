#!/bin/bash
#SBATCH --partition=single,lattice,parallel,theia,cpu2013,cpu2019 
#SBATCH --time=7-0:0:0
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=1 
#SBATCH --mem=16GB
#SBATCH --job-name=XAE4Exp-job.slurm
#SBATCH --out=%x-%N-%j.out
#SBATCH --error=%x-%N-%j.error

source $HOME/.bash_profile
conda activate XAI

python src/GeneSelectionByPathway.py 104 ./data/ ./data/pathways/Pathways.csv ./data/filter/input ./data/filter/model $SLURM_JOB_ID > test.log
