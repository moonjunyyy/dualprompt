import sys

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

from models.L2P import L2P
from models.dualprompt import DualPrompt
from utils.argvs import l2p_argvs
from utils.trainer_continual import trainer_til

def main(**kwargs):

    MODEL_PATH     = kwargs["--save-path"]
    DATA_PATH      = kwargs["--data-path"]

    batchsize      = int(kwargs["--batchsize"])
    step_size      = int(kwargs["--stepsize"])
    batch_per_step = step_size // batchsize
    backbone_name  = kwargs["--backbone-name"]
    epochs         = int(kwargs["--epochs"])
    log_freqency   = int(kwargs["--log-freqency"])
    pool_size      = int(kwargs["--pool-size"])
    selection_size = int(kwargs["--selection-size"])
    prompt_len     = int(kwargs["--prompt-len"])
    dimention      = int(kwargs["--dimention"])
    num_tasks      = int(kwargs["--num-tasks"])
    num_class      = int(kwargs["--num-class"])
    lr_scheduler   = kwargs["--lr-scheduler"]
    use_amp        = kwargs["--use-amp"]
    debug          = kwargs["--debug"]
    seed           = kwargs["--seed"]

    if seed is not None:
        seed = int(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    transformCifar = transforms.Compose([transforms.Resize   (224),
                                         transforms.ToTensor ()])
    train_dataset = CIFAR100(DATA_PATH, download=True, train=True,  transform=transformCifar)
    test_dataset  = CIFAR100(DATA_PATH, download=True, train=False, transform=transformCifar)

    train = trainer_til( model           = DualPrompt,
                         model_args      =
                            {"dimention"     : dimention,
                             "task_num"      : num_tasks,
                             "pos_g_prompt"  : (1,2),
                             "len_g_prompt"  : 5,
                             "pos_e_prompt"  : (3,4,5),
                             "len_e_prompt"  : 20,
                             "prompt_func"   : "prefix_tuning",
                             "class_num"     : num_class,
                             "backbone_name" : backbone_name},
                         train_dataset   = train_dataset,
                         test_dataset    = test_dataset,
                         batch_size      = batchsize,
                         epochs          = epochs,
                         num_tasks       = num_tasks,
                         step_size       = 1,
                         log_freqency    = log_freqency,
                         save_dir        = MODEL_PATH,
                         optimizer       = optim.Adam,
                         optimizer_args  =
                            {"lr"    : 0.005,
                             "betas" : (0.9, 0.999)},
                         lr_scheduler    = None,
                         lr_schedul_args = None,
                         use_amp         = use_amp,
                         debug           = debug)
    train.train()

if __name__ == "__main__":
    for n, name in enumerate(sys.argv):
        try:
            if n % 2 == 0:
                continue
            l2p_argvs[name] = sys.argv[n+1]
        except KeyError:
            print("Unknown argument: {}, pass".format(name)) 

    if   l2p_argvs["--backbone-name"].find('base')  != -1:
        l2p_argvs ["--dimention"] = 768
    elif l2p_argvs["--backbone-name"].find('tiny')  != -1:
        l2p_argvs ["--dimention"] = 192
    elif l2p_argvs["--backbone-name"].find('small') != -1:
        l2p_argvs ["--dimention"] = 384
    elif l2p_argvs["--backbone-name"].find('large') != -1:
        l2p_argvs ["--dimention"] = 1024
    else:
        print("Unknown backbone name: {}".format(l2p_argvs["--backbone-name"]))
        exit(1)
    
    main(**l2p_argvs)
    print("Done")