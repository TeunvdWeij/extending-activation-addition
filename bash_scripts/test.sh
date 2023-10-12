#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=rome
#SBATCH --time=00:05:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

#Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

#Execute a Python program
python src/skip_tokens.py
