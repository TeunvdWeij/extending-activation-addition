#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=rome
#SBATCH --time=00:05:00
 
#Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0d
 
#Execute a Python program 
python src/skip_tokens.py
