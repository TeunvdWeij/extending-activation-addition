#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --mem=40GB
#SBATCH --output=job_output/get_acts_job_%j.out
#SBATCH --error=job_output/get_acts_job_%j.err

#Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

source output_control_venv/bin/activate
python src/generate_activations.py --no-mean --mode "only_code" --version "2.17" --note 