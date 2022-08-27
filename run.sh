#!/bin/bash

#SBATCH --job-name=moon_L2P
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH -o output.out
#SBATCH -e errors.err

source /home/junyeong/init.sh
conda activate dualprompt
python main.py \
    --data-path "/home/datasets/CIFAR-100/cifar-100-python" \
    --save-path "saved/l2p/CIL_NoDiversed_1055" \
    --pool-size 10 \
    --selection-size 5 \
    --prompt-len 5 \
    --batchsize 128