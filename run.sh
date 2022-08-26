#!/bin/bash

#SBATCH --job-name=moon
#SBATCH --node=1
#SBATCH --gres=gpu:1  
#SBATCH --time=5:00:00
#SBATCH -o output.out
#SBATCH -e errors.err

source /home/junyeong/init.sh
conda activate dualprompt
python main.py