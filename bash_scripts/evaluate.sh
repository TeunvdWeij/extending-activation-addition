#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --time=00:45:00
#SBATCH --output=job_output/eval_%j.out
#SBATCH --error=job_output/eval_%j.err

#Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0

source output_control_venv/bin/activate
# python src/evaluate.py --version 6.01 --note "manual ics!" --ics 0.0 0.1 0.2 0.3 0.5 0.6 0.7 0.8 0.85 0.9 1 1.1 1.2 1.4 1.5 1.8 2 5 10 20 50 --layers 0 --neg_acts only_code --pos_acts only_text --modes only_code only_text
# python src/evaluate.py --version 6.02 --note "manual ics!" --ics 0 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 30 50 100 --layers 5 --neg_acts only_code --pos_acts only_text --modes only_code only_text
# python src/evaluate.py --version 6.03 --note "manual ics!" --ics 0 10 12.5 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 40 50 100 200 --layers 10 --neg_acts only_code --pos_acts only_text --modes only_code only_text
# python src/evaluate.py --version 6.04 --note "manual ics!" --ics 0 10 15 16 17.5 19 20 22.5 25 27.5 30 32.5 35 37.5 40 45 50 55 60 65 70 100 200 --layers 15 --neg_acts only_code --pos_acts only_text --modes only_code only_text
# python src/evaluate.py --version 6.05 --note "manual ics!" --ics 0 10 15 17.5 20 22.5 25 27.5 30 32.5 35 37.5 40 42.5 45 47.5 50 52.5 55 57.5 60 65 70 80 100 200 500 --layers 20 --neg_acts only_code --pos_acts only_text --modes only_code only_text
# python src/evaluate.py --version 6.06 --note "manual ics!" --ics 0 15 20 25 27.5 30 32.5 35 40 45 50 55 60 65 70 75 80 90 100 125 175 250 500 --layers 25 --neg_acts only_code --pos_acts only_text --modes only_code only_text
python src/evaluate.py --version 6.07 --note "manual ics!" --ics 0 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 110 120 130 140 150 175 200 250 500 1000  --layers 30 --neg_acts only_code --pos_acts only_text --modes only_code only_text