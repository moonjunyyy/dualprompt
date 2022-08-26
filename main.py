import os
import sys

import torch.optim as optim
import torchvision.transforms as transforms
from   torchvision.datasets import CIFAR100

from models.L2P import L2P
from utils.trainer_continual import trainer_til

MODEL_PATH     = "saved/l2p/CIL2"
DATA_PATH      = "/home/datasets/CIFAR-100/cifar-100-python"

batchsize      = 128
step_size      = 128
batch_per_step = step_size // batchsize
backbone_name  = "vit_base_patch16_224"
epochs         = 5
log_interval   = 20
pool_size      = 10
selection_size = 4
prompt_len     = 10
dimention      = 768
num_tasks      = 10
num_class      = 100
lr_scheduler   = None
use_amp        = True

def main():
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
    main()
    print("Done")