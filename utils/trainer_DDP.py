import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from utils.trainer import trainer


class trainer_DDP(trainer):
    def __init__(self,
                 model          : nn.Module         = None,
                 model_args     : dict              = None,
                 train_dataset  : Dataset           = None,
                 test_dataset   : Dataset           = None,
                 batch_size     : int               = 16,
                 num_workers    : int               = 2,
                 epoch_start    : int               = 1,
                 epochs         : int               = 100,
                 step_size      : int               = 1,
                 log_freqency   : int               = 10,
                 save_dir       : str               = None,
                 optimizer      : optim.Optimizer   = None,
                 optimizer_args : dict              = None,
                 lr_scheduler   : _LRScheduler      = None,
                 lr_schedul_args: dict              = None,
                 use_amp        : bool              = False,
                 debug          : bool              = False,
                 **kwargs) -> None:
        r'''
        Initialize the trainer.

        Args:
            model          : The model to be trained. 
            train_dataset  : The training dataset.  
            test_dataset   : The testing dataset.
            batch_size     : The batch size. Default: 16.
            num_workers    : The number of workers to be used. Default: 2.
            epoch_start    : The epoch to start from. Default: 1.
            epochs         : The number of epochs to train for. Default: 100.
            step_size      : The step size of the optimizer in number of batch. Default: 1.
            log_freqency   : The frequency of logging per epoch. Default: 10.
            save_dir       : The directory to save the model. None to disable.
            optimizer      : The optimizer to be used. Default: SGD.
            lr_scheduler   : The learning rate scheduler. Default: Constant.
            use_amp        : Whether to use amp. Default: False.
        '''
        if model is None:
            raise ValueError("The model is not specified.")
        if train_dataset is None or test_dataset is None:
            raise ValueError("train_dataset or test_dataset is not specified.")

        self.model_fn         = model
        self.model_args       = model_args

        self.optimizer_fn     = optim.SGD if optimizer is None else optimizer
        self.optimizer_args   = optimizer_args if optimizer_args is not None else {}
        self.lr_scheduler_fn  = optim.lr_scheduler.ConstantLR if lr_scheduler is None else lr_scheduler
        self.lr_schedul_args  = lr_schedul_args if lr_schedul_args is not None else {}
        
        self.batch_size       = batch_size
        self.epoch            = epoch_start
        self.epochs           = epochs
        self.step_size        = step_size * batch_size
        self.log_freqency     = log_freqency
        self.use_amp          = use_amp
        self.scaler           = torch.cuda.amp.GradScaler(enabled = use_amp)
        self.save_dir         = save_dir

        self.train_dataset    = train_dataset
        self.test_dataset     = test_dataset

        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True)
        self.test_dataloader  = DataLoader(test_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=False)

        self._counts  = 0
        self._metrics = {}

        if save_dir is not None:
            self.load()
            self.writer       = SummaryWriter(log_dir = save_dir)
        else:
            self.writer       = None
        self._debug = debug

    def _initialize_model(self, rank, world_size, **kwargs):
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
        self.model = self.model_fn(**self.model_args).to(rank)
        self.model = DDP(self.model, device_ids=[rank])
        self.optimizer = self.optimizer_fn(self.model.parameters(), **self.optimizer_args)
        self.optim_init_dict = self.optimizer.state_dict()
        self.lr_scheduler = self.lr_scheduler_fn(self.optimizer, **self.lr_schedul_args)
        return

    def train(self, rank, world_size, **kwargs):
        r'''
        Train the model.
        '''
        self._initialize_model(rank, world_size, **kwargs)

        for self.epoch in range(1, self.epochs + 1):
            self._set_writer("Train/")
            self._train_a_epoch(self.train_dataloader)
            self._set_writer("Test/")
            self._test_a_epoch (self.test_dataloader)
        return    