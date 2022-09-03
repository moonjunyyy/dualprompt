import os
import sys
import time

import torch
import torch.optim as optim
import torch.multiprocessing as mp

from models.L2P import L2P
from helper.log import Log
from helper.argvs import log_parser, parse_args
from utils.trainer import trainer

def main(args):
    print(args)
    _log_op, _ = log_parser.parse_known_args(sys.argv)
    print(_log_op)
    Log.log_init(_log_op)
    train = trainer(args)
    train.run()
    print(Log._Log)
    time.sleep(60)

if __name__ == "__main__":
    print(sys.argv)
    main(parse_args(sys.argv))
    print("Done")
