import os
import sys
import time

import torch
import torch.optim as optim
import torch.multiprocessing as mp

from models.L2P import L2P
from helper.log import Log
from helper.argvs import log_parser, parse_args
from utils.trainer import image_trainer

os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"

def main(args):
    print(args)

    image_trainer(args)
    time.sleep(60)

if __name__ == "__main__":
    print(sys.argv)
    main(parse_args(sys.argv))
    print("Done")
