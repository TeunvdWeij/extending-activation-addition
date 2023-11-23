#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --output=job_output/eval_%j.out
#SBATCH --error=job_output/eval_%j.err

#Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

source output_control_venv/bin/activate
python src/evaluate.py --version 3.07 --note "all layers, minimal ics" --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 --pos_acts only_text --neg_acts only_code all --ics 0 1 2 5 8 10 12 15 20 30 50