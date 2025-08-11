#!/bin/bash
#SBATCH -t 00:50:00
#SBATCH --gres shard:1
#SBATCH --constraint any-A100
#SBATCH --partition abugoot
#SBATCH --mem 10GB
#SBATCH -o logs/o_analysis_%a
#SBATCH -e logs/e_analysis_%a
#SBATCH --array=0-3

# Define dataset names and corresponding batch sizes
datasets=("GoM" "LV" "pbmc" "Repressilator")

# Get the dataset for this array task
dataset_name=${datasets[$SLURM_ARRAY_TASK_ID]}

python results_analysis_clean.py --dataset ${dataset_name}