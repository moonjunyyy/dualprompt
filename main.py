import os
import sys

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

from models.L2P import L2P
from utils.argvs import l2p_argvs
from utils.trainer_continual import trainer_til

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(**kwargs):

    MODEL_PATH     = kwargs["--save-path"]
    DATA_PATH      = kwargs["--data-path"]

    batchsize      = kwargs["--batchsize"]
    step_size      = kwargs["--stepsize"]
    batch_per_step = step_size // batchsize
    backbone_name  = kwargs["--backbone-name"]
    epochs         = kwargs["--epochs"]
    log_interval   = kwargs["--log-interval"]
    pool_size      = kwargs["--pool-size"]
    selection_size = kwargs["--selection-size"]
    prompt_len     = kwargs["--prompt-len"]
    dimention      = kwargs["--dimention"]
    num_tasks      = kwargs["--num-tasks"]
    num_class      = kwargs["--num-class"]
    lr_scheduler   = kwargs["--lr-scheduler"]
    use_amp        = kwargs["--use-amp"]

    transformCifar = transforms.Compose([transforms.Resize   (224),
                                         transforms.ToTensor ()])

    train_dataset = CIFAR100(DATA_PATH, download=True, train=True,  transform=transformCifar)
    test_dataset  = CIFAR100(DATA_PATH, download=True, train=False, transform=transformCifar)

    #config = resolve_data_config({}, model=backbone)
    #transform = create_transform(**config)
    model = L2P(pool_size      = pool_size,
                selection_size = selection_size,
                prompt_len     = prompt_len,
                dimention      = dimention,
                class_num      = num_class,
                backbone_name  = backbone_name)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr = 0.03, betas=(0.9, 0.999))
    train = trainer_til(model,
                        optimizer,
                        train_dataset,
                        test_dataset,
                        num_tasks    = num_tasks,
                        epochs       = epochs,
                        batch_size   = batchsize,
                        step_size    = batch_per_step,
                        log_interval = log_interval,
                        save_dir     = MODEL_PATH,
                        use_amp      = use_amp)
    train.cuda()
    train.train()

if __name__ == "__main__":
    for n, name in enumerate(sys.argv):
        print(n, name)
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
