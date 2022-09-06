import os
import sys
import time

from helper.argvs import parse_args
from utils.trainer import image_trainer

os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main(args):

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '60020'

    print(args)
    image_trainer(args)
    time.sleep(60)

if __name__ == "__main__":
    main(parse_args(sys.argv))
    print("Done")
