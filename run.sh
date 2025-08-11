#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH --gres shard:1
#SBATCH --constraint any-A100
#SBATCH --partition abugoot
#SBATCH --mem 1GB
#SBATCH -o logs/o_forecast_%a
#SBATCH -e logs/e_forecast_%a
#SBATCH --array=0-39

# Define the list of seeds
seeds=(1 2 3 4 5 40 41 42 43 44)

# Define the four python commands
commands=(
    "python experiments/realdata/realdata.py pbmc"
    "python experiments/realdata/realdata.py GoM"
    "python experiments/classic/classic_sde.py LV"
    "python experiments/classic/classic_sde.py Repressilator"
)

# Calculate which script and seed to use based on SLURM_ARRAY_TASK_ID
script_idx=$((SLURM_ARRAY_TASK_ID / 10))
seed_idx=$((SLURM_ARRAY_TASK_ID % 10))

# Get the specific seed and command
seed=${seeds[$seed_idx]}
command=${commands[$script_idx]}

# Run the command with the specific seed
$command $seed