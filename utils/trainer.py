import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


class trainer():
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

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.device       = torch.device("cuda")
            self.model        = nn.DataParallel(model(device = self.device, **model_args).to(self.device), range(0, torch.cuda.device_count()))
            self.useMultiGPU  = True
        else :
            self.device       = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.model        = model(device = self.device, **model_args).to(self.device)
            self.useMultiGPU  = False

        self.optimfn = optim.SGD if optimizer is None else optimizer
        if optimizer is None : self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        else:                  self.optimizer = optimizer(self.model.parameters(), **optimizer_args)
        self.optim_init_dict  = self.optimizer.state_dict()

        if lr_scheduler  is None : self.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, 1.0)
        else:                      self.lr_scheduler = lr_scheduler(self.optimizer, **lr_schedul_args)

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
                                           shuffle=True, pin_memory_device=self.device)
        self.test_dataloader  = DataLoader(test_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=False, pin_memory_device=self.device)

        self._counts  = 0
        self._metrics = {}

        if save_dir is not None:
            self.load()
            self.writer       = SummaryWriter(log_dir = save_dir)
            self.tag_counter  = {}
        else:
            self.writer       = None
        self._debug = debug



    def train(self, **kwargs):
        r'''
        Train the model.
        '''
        for self.epoch in range(1, self.epochs + 1):
            self._set_writer("Train/")
            self._train_a_epoch(self.train_dataloader)
            self._set_writer("Test/")
            self._test_a_epoch (self.test_dataloader)
        return

    def _train_a_epoch(self, **kwargs):
        self.model.train()
        length   = len(self.train_dataloader)
        interval = length // self.log_freqency
        for n, batch in enumerate(self.train_dataloader):
            with torch.cuda.amp.autocast(enabled=self.use_amp) :
                inputs, targets = batch
                inputs  = inputs .to(self.device)
                targets = targets.to(self.device)
                inference = self.model(inputs)

                if self.useMultiGPU:
                    metrics   = self.model.module.metrics (inference, targets)
                else:
                    metrics   = self.model.metrics (inference, targets)

                self.scaler.scale(metrics['loss']).backward()
                self._set_metrics(**metrics)

                if n % self.step_size == self.step_size - 1 or n == length - 1:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                if self._debug:
                    self._print_log( "Train ", self.train_dataloader, n, True)
                if (n % interval) == interval - 1       or n == length - 1:
                    self._print_log( "Train ", self.train_dataloader, n, False)
                    if self.writer is not None:
                        self._add_scalar("Train")
        print("")
        if self.save_dir is not None:
            self.save()
        self.lr_scheduler.step()
        self._reset_metrics()
        return

    def _test_a_epoch(self, **kwargs):
        self.model.eval()
        with torch.no_grad():
            for n, batch in enumerate(self.test_dataloader):

                inputs, targets = batch
                inputs  = inputs .to(self.device)
                targets = targets.to(self.device)
                inference = self.model(inputs)

                if self.useMultiGPU:
                    metrics   = self.model.module.metrics (inference, targets)
                else:
                    metrics   = self.model.metrics (inference, targets)
                self._set_metrics(**metrics)
                if self._debug:
                    self._print_log("Test ", self.test_dataloader, n, True)

        self._print_log("Test ", self.test_dataloader, n, False)
        if self.writer is not None:
            self._add_scalar("Test")
        self._reset_metrics()
        return

    def _print_log(self, name, dataloader, n, debug = False, **kwawrgs):

        if debug:
            print('{} Epoch: {:3d} [{}/{:3d} ({:.0f}%)]'.format(name, self.epoch, 
                (n + 1) *  self.batch_size if (n + 1) *  self.batch_size < len(dataloader.dataset) else len(dataloader.dataset),
                len(dataloader.dataset), 100. * (n + 1) / len(dataloader)),end = "")
            for k,v in self._metrics.items():
                print("\t{}: {:.6f}".format(k, v / self._counts), end = "")
            print("", end="\r")

        else:
            print('{} Epoch: {} [{}/{} ({:.0f}%)]'.format(name, self.epoch, 
                (n + 1) *  self.batch_size if (n + 1) *  self.batch_size < len(dataloader.dataset) else len(dataloader.dataset),
                len(dataloader.dataset), 100. * (n + 1) / len(dataloader)),end = "")
            for k,v in self._metrics.items():
                print("\t{}: {:.6f}".format(k, v / self._counts), end = "")
            print("")        
        return

    def _set_writer(self, name, **kwargs):
        if self.writer is not None:
            self.writer = SummaryWriter(os.path.join(self.save_dir, name))
        return

    def _set_metrics(self, **kwargs):
        r'''
        Log the metrics of the model.
        '''
        self._counts += 1
        for k, v in kwargs.items():
            try:
                self._metrics[k] += v
            except:
                self._metrics[k]  = v
        return

    def _reset_metrics(self, **kwargs):
        r'''
        Reset the metrics of the model.
        '''
        self._counts = 0
        for k, v in self._metrics.items():
            self._metrics[k] = 0
        return

    def _add_scalar(self, tag : str, **kwargs):
        r'''
        Set the writer of the model.
        '''
        try:
            i = self.tag_counter[tag]
        except:
            self.tag_counter[tag] = 0

        for k,v in self._metrics.items():
            self.writer.add_scalar(tag + '/' + k, v / self._counts, self.tag_counter[tag])
        self.tag_counter[tag] += 1
        return

    def save(self, **kwargs):
        r'''
        Save the model.
        '''
        torch.save({
            'epoch'                  : self.epoch,
            'model_state_dict'       : self.model.state_dict(),
            'optimizer_state_dict'   : self.optimizer.state_dict(),
            'scaler_state_dict'      : self.scaler.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'metrics'                : self._metrics,
            'step'                   : self._counts,
        }, os.path.join(self.save_dir, 'checkpoint_{}.pth'.format(self.epoch)))
        return

    def load(self, *args, **kwargs):
        r'''
        Load the model.
        '''
        try:
            for e in range(self.epochs + 1):
                load_dict = torch.load(os.path.join(self.save_dir, 'checkpoint_{}.pth'.format(e)))
        except:
            pass
        try:
            self.model.load_state_dict       (load_dict['model_state_dict'])
            self.optimizer.load_state_dict   (load_dict['optimizer_state_dict'])
            self.scaler.load_state_dict      (load_dict['scaler_state_dict'])
            self.lr_scheduler.load_state_dict(load_dict['lr_scheduler_state_dict'])
            self.epoch      = load_dict['epoch']
            self._metrics   = load_dict['metrics']
            self._counts      = load_dict['step']
        except:
            pass
        return
    