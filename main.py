import os
import sys

import torch
import torch.optim as optim
import torch.multiprocessing as mp

from models.L2P import L2P
from helper.argvs import parser, parse_args
from utils.trainer import trainer

def main(**kwargs):
    train = trainer(**kwargs)
    train.run()

if __name__ == "__main__":
    print(parse_args(sys.argv))
    main(**parse_args(sys.argv))
    print("Done")
