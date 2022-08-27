import os

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

from utils.trainer import trainer


class trainer_til(trainer):
    def __init__(self,
                 model          : nn.Module         = None,
                 train_dataset  : Dataset           = None,
                 test_dataset   : Dataset           = None,
                 batch_size     : int               = 16,
                 num_tasks      : int               = 1,
                 num_workers    : int               = 2,
                 epoch_start    : int               = 1,
                 epochs         : int               = 100,
                 step_size      : int               = 1,
                 log_freqency   : int               = 10,
                 save_dir       : str               = None,
                 optimizer      : optim.Optimizer   = None,
                 lr_scheduler   : _LRScheduler      = None,
                 use_amp        : bool              = False,
                 **kwargs) -> None:
                  
        r'''
        Initialize the trainer class.
        
        Parameters
        --------------------------------------------------------------
        model : nn.Module
            The model to be trained.
            accuracy and loss_fn needed to be implemented.
        optimizer : torch.optim.Optimizer
            The optimizer to be used.
        train_dataset : Dataset
            The dataset for training.
        test_dataset : Dataset
            The dataset for testing.
        epochs : int
            The number of epochs to train for.
        step_size : int
            The step size for the optimization.
        log_interval : int
            The number of batches to wait before logging.
        save_dir : str
            The directory to save the model.
        lr_scheduler : torch.optim.lr_scheduler._LRScheduler
            The learning rate scheduler.
        '''
        super().__init__(model,
                         train_dataset,
                         test_dataset,
                         batch_size,
                         num_workers,
                         epoch_start,
                         epochs,
                         step_size,
                         log_freqency,
                         save_dir,
                         optimizer,
                         lr_scheduler,
                         use_amp,
                         **kwargs)

        self.num_workers          = num_workers
        self.num_tasks            = num_tasks
        self._class_per_task      = self._class_split()
        self.save_dir             = save_dir

        print('\nClass per task : ', self._class_per_task, end="\n\n")
        self._train_data_per_task = [Subset(self.train_dataset, self._dataset_mask(cls, self.train_dataset)) for cls in self._class_per_task]
        self._test_data_per_task  = [Subset(self.test_dataset,  self._dataset_mask(cls, self.test_dataset )) for cls in self._class_per_task]

        self.taskID = 0
        self.testID = 0

        self.train_dataloader = DataLoader(self._train_data_per_task[self.taskID],
                                           batch_size  = self.batch_size,
                                           shuffle     = True,
                                           num_workers=num_workers)

        self.test_dataloader  = DataLoader(self._test_data_per_task [self.testID],
                                           batch_size  = self.batch_size,
                                           shuffle     = False,
                                           num_workers=num_workers)
        
    def _convert_task(self, task_idx = -1, test_idx = -1):
        r'''
        Convert the task index to the task index of the next epoch.
        '''
        if task_idx != -1:
            self.taskID = task_idx
        if test_idx != -1:
            self.testID = test_idx
        self.train_dataloader = DataLoader(self._train_data_per_task[self.taskID],
                                           batch_size  = self.batch_size,
                                           shuffle     = True,
                                           num_workers = self.num_workers)

        self.test_dataloader  = DataLoader(self._test_data_per_task [self.testID],
                                           batch_size  = self.batch_size,
                                           shuffle     = False,
                                           num_workers = self.num_workers)

    def _class_split(self):
        r'''
        Split the dataset into num_tasks classes.
        '''
        num_classes = len(self.train_dataset.classes)
        classes = torch.arange(num_classes)
        class_split = []
        class_num   = int(num_classes // self.num_tasks)
        cursor = 0
        for i in range(self.num_tasks):
            class_split.append(classes[cursor:cursor+class_num])
            cursor += class_num
        return class_split

    def _class_split(self):
        r'''
        Split the dataset into num_tasks classes.
        '''
        num_classes = len(self.train_dataset.classes)
        classes = torch.randperm(num_classes)
        class_split = []
        for i in range(self.num_tasks):
            class_split.append(classes[i::self.num_tasks])
        return class_split

    def _dataset_mask(self,
                      cls     : list,
                      dataset : torch.utils.data.Dataset,
                      *args, **kwargs) -> list:
        mask = (torch.tensor(dataset.targets) == cls.unsqueeze(-1)).sum(dim = -2).nonzero().squeeze()
        return mask

    def _train_one_task(self, task_idx):
        r'''
        Train one task.
        '''
        self.model.train()
        self._convert_task(task_idx, 0)
        for self.epoch in range(1, self.epochs + 1): 
            self._train_a_epoch()

    def _test_one_task (self, task_idx):
        r'''
        Test for past tasks.
        '''
        for task in range(0, task_idx + 1) :
            print("Task : ", task)
            self.model.eval()
            self._convert_task(test_idx = task)
            self._test_a_epoch()
        print("")

    def train(self, *args, **kwargs):
        for task_idx in range(self.num_tasks):
            print("Train for : ", task_idx)

            self._set_writer("Train/Task{}".format(task_idx))
            self.model.task_mask(self._class_per_task[task_idx])
            self._train_one_task(task_idx)
            
            plt.bar(range(len(self.model.prompt.update().cpu().numpy())), self.model.prompt.update().cpu().numpy())
            plt.savefig(self.save_dir + '/' + 'selection'+ str(task_idx) +'.png')
            plt.clf()
            
            self._set_writer("Test/Task{}".format(task_idx))
            self._test_one_task (task_idx)
