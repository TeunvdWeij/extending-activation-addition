#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --mem=40GB
#SBATCH --output=job_output/grid_search_%j.out
#SBATCH --error=job_output/grid_search_%j.err

#Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

source output_control_venv/bin/activate
python src/multi_steering_grid_search.py --version "0.03" --sign pos --note "trying out multi steering activations, few validation prompts" --multipliers 0 0.5 1 5
