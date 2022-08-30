import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100, CIFAR10, MNIST, FashionMNIST
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

class trainer():
    def __init__(self,
                 model          : nn.ModuleDict,
                 model_args     : dict,
                 criterion      : nn.Module,
                 optimizer      : optim.Optimizer,
                 optimizer_args : dict,
                 scheduler      : _LRScheduler,
                 scheduler_args : dict,
                 worldsize      : int,
                 batchsize      : int,
                 step_size      : int,
                 epochs         : int,
                 log_freqency   : int,
                 num_tasks      : int,
                 task_governor  : str,
                 dataset        : str,
                 dataset_path   : str,
                 num_workers    : int,
                 save_path      : str,
                 use_amp        : bool,
                 debug          : bool,
                 **kwargs) -> None:
        
        self.model          = model
        self.model_args     = model_args
        self.criterion      = criterion
        self.optimizer      = optimizer
        self.optimizer_args = optimizer_args
        self.scheduler      = scheduler
        self.scheduler_args = scheduler_args
        self.worldsize      = worldsize
        self.batchsize      = batchsize
        self.step_size      = step_size
        self.epochs         = epochs
        self.log_freqency   = log_freqency
        self.num_tasks      = num_tasks
        self.task_governor  = task_governor
        self.dataset        = self._dataset(dataset)
        self.dataset_path   = dataset_path
        self.num_workers    = num_workers
        self.save_path      = save_path
        self.use_amp        = use_amp
        self.debug          = debug
        
        #setting up the distributed environment
        self._init_device()

        #setting up the dataset
        transform = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor ()])
        train_dataset = self.dataset(dataset_path, download=True, train=True,  transform=transform)
        test_dataset  = self.dataset(dataset_path, download=True, train=False, transform=transform)
        
        return

    def __call__(self, *args, **kwargs):
        return self.train(*args, **kwargs)

    def train(self, gpu, *args, **kwargs):
        if self.distributed:
            world_size = self.worldsize * self.ngpus_per_node
            mp.spawn(self._worker, nprocs=world_size, args=(gpu,))
        else:
            self._worker(gpu, self.ngpus_per_node, *args, **kwargs)

    def _worker(self, gpu, ngpus_per_node, *args, **kwargs):
        
        return
    def _init_device(self):
        if self.worldsize == -1: self.worldsize=int(os.environ['WORLD_SIZE'])
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.mode = 'ddp'
        else: self.mode = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.distributed = self.world_size > 1
        self.ngpus_per_node = torch.cuda.device_count()

    def _dataset(self, _data : str) -> Dataset:
        if _data == 'CIFAR100':
            self.classes = 100
            return CIFAR100
        elif _data == 'CIFAR10':
            self.classes = 10
            return CIFAR10
        elif _data == 'MNIST':
            self.classes = 10
            return MNIST
        elif _data == 'FashionMNIST':
            self.classes = 10
            return FashionMNIST
        else:
            raise ValueError('Dataset {} not supported'.format(_data))
