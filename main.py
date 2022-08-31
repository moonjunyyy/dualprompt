from cmath import log
import os
import sys

import torch
import torch.optim as optim
import torch.multiprocessing as mp

from models.L2P import L2P
from helper.log import Log
from helper.argvs import log_parser, parse_args
from utils.trainer import trainer

def main(**kwargs):
    _log_op = vars(log_parser.parse_known_args(kwargs)[0])
    Log.log_init(**_log_op)
    train = trainer(**kwargs)
    train.run()

if __name__ == "__main__":
    main(**parse_args(sys.argv))
    print("Done")
