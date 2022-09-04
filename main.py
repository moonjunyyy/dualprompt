import os
import sys
import time

from helper.argvs import parse_args
from utils.trainer import image_trainer

os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"

def main(args):
    print(args)
    image_trainer(args)
    time.sleep(60)

if __name__ == "__main__":
    main(parse_args(sys.argv))
    print("Done")
