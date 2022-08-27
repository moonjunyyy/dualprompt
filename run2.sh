#!/bin/bash

#SBATCH --job-name=moon_L2P2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH -o output_ND2.out
#SBATCH -e errors_ND2.err

source /home/junyeong/init.sh
conda activate dualprompt
python main.py \
    --data-path "/home/datasets/CIFAR-100/cifar-100-python" \
    --save-path "saved/l2p/CIL_NoDiversed_10410" \
    --pool-size 10 \
    --selection-size 4 \
    --prompt-len 10 \
    --batchsize 128