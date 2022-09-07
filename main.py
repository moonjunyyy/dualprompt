import os
import sys
import time

from helper.argvs import parse_args
from utils.trainer import Imgtrainer

os.environ["TORCH_DISTRIBUTED_DEBUG"]="DETAIL"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main(kwargs):
    print(kwargs)
    trainer = Imgtrainer(**kwargs)
    trainer.run()
    time.sleep(60)

if __name__ == "__main__":
    main(vars(parse_args(sys.argv)))
    print("Done")
