import os
import random
import time
from enum import Enum
from typing import Callable

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

from helper.log import Log

########################################################################################################################
# This is trainer with a DistributedDataParallel Based on the following tutorial:                                      #
# https://github.com/pytorch/examples/blob/main/imagenet/main.py                                                       #
########################################################################################################################

class trainer():
    def __init__(self,
                 model          : nn.ModuleDict,
                 model_args     : dict,
                 criterion      : nn.Module | Callable,
                 optimizer      : optim.Optimizer,
                 optimizer_args : dict,
                 scheduler      : _LRScheduler,
                 scheduler_args : dict,
                 batchsize      : int,
                 stepsize       : int,
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
                 seed           : int,
                 worldsize      : int,
                 training       : bool = True,
                 rank           : int = 2,
                 multiprocessing_distributed : bool = True,
                 dist_url                    : str  = "env://",
                 dist_backend                : str  = "nccl",
                 gpu                         : int  = None,
                 **kwargs) -> None:
        mp.set_start_method('fork')
        self.model_fn       = model
        self.model_args     = model_args
        self.criterion_fn   = criterion
        self.optimizer_fn   = optimizer
        self.optimizer_args = optimizer_args
        self.scheduler_fn   = scheduler
        self.scheduler_args = scheduler_args

        self.worldsize      = worldsize
        self.batchsize      = batchsize
        self.stepsize       = stepsize
        self.epoch          = 1
        self.epochs         = epochs
        self.training       = training

        self.log_freqency   = log_freqency

        self.num_tasks      = num_tasks
        self.task_governor  = task_governor
        if self.task_governor is not None:
            Log.log_warning('Task governor is not implemented yet. Ignore the keyword and works CIL setting.')
        self.dataset        = self._dataset(dataset)
        self.dataset_path   = dataset_path

        self.num_workers    = num_workers
        self.save_path      = save_path
        self.use_amp        = use_amp
        self.debug          = debug
        self.seed           = seed

        self.dist_url       = dist_url
        self.dist_backend   = dist_backend
        self.gpu            = gpu

        self.multiprocessing_distributed = multiprocessing_distributed
        self.training = True
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            cudnn.deterministic = True
            Log.log_info('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')
            pass

        
        transform = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor ()])
        self.train_dataset   = self.dataset(self.dataset_path, download=True, train=True,  transform=transform)
        self.test_dataset    = self.dataset(self.dataset_path, download=True, train=False, transform=transform)
        self._class_per_task = self._class_split()
        self._task_id        = 0
        self._test_id        = 0
        self.log_interval    = 0

    def __call__(self, **kwargs):
        return self.run(**kwargs)

    def run(self): 
        if self.worldsize == -1 : self.worldsize = os.environ["WORLD_SIZE"] # Hard coded for now
        self.distributed  = self.worldsize > 1 or self.multiprocessing_distributed
        ngpus_per_node    = torch.cuda.device_count()
        if self.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total worldsize
            # needs to be adjusted accordingly
            self.worldsize = ngpus_per_node * self.worldsize
            if self.worldsize == 0:
                Log.log_exception('WORLD_SIZE is not set correctly. Please set it to the number of GPUs in your system.')
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # _worker process function
            mp.spawn(self._worker, nprocs=ngpus_per_node, args=(ngpus_per_node))
        else:
            # Simply call _worker function
            self._worker(self.gpu, ngpus_per_node)
        return

    def _convert_task(self, task_idx):
        r'''
        Convert the task index to the task index of the next epoch.
        '''
        if task_idx != -1:
            self.taskID = task_idx
        train = Subset(self.train_dataset, self._dataset_mask(self._class_per_task[task_idx], self.train_dataset))
        if self.distributed:
            train_sampler = DistributedSampler(train)
        else:   
            train_sampler = None
            test_sampler  = None

        self.train_loader = DataLoader(
            train, batch_size=self.batchsize, shuffle=(train_sampler is None),
            num_workers=self.num_workers, pin_memory=True, sampler=train_sampler)

    def _convert_test(self, test_idx):
        r'''
        Convert the task index to the task index of the next epoch.
        '''
        self.testID = test_idx
        test  = Subset(self.test_dataset,  self._dataset_mask(self._class_per_task[test_idx], self.test_dataset))
        if self.distributed:
            test_sampler  = DistributedSampler(test, shuffle=False, drop_last=True)
        else:   
            test_sampler  = None
        self.test_loader = DataLoader(
            test, batch_size=self.batchsize, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, sampler=test_sampler)

    def _class_split(self, random : bool = True):
        r'''
        Split the dataset into num_tasks classes.
        '''
        if random:
            num_classes = len(self.train_dataset.classes)
            classes = torch.randperm(num_classes)
            class_split = []
            for i in range(self.num_tasks):
                class_split.append(classes[i::self.num_tasks])
            return class_split
        else:
            num_classes = len(self.train_dataset.classes)
            classes = torch.arange(num_classes)
            class_split = []
            class_num   = int(num_classes // self.num_tasks)
            cursor = 0
            for i in range(self.num_tasks):
                class_split.append(classes[cursor:cursor+class_num])
                cursor += class_num
            return class_split

    def _dataset_mask(self,
                      cls     : list,
                      dataset : torch.utils.data.Dataset,
                      **kwargs) -> list:
        mask = (torch.tensor(dataset.targets) == cls.unsqueeze(-1)).sum(dim = -2).nonzero().squeeze()
        return mask

    def _worker(self, gpu, ngpus_per_node):
        self.ngpus_per_node = torch.cuda.device_count()
        if gpu is not None:
            Log.log('Using GPU:{} for training.'.format(self.gpu))

        if self.distributed:
            if self.dist_url == "env://" and self.rank == -1:
                self.rank = int(os.environ["RANK"])
            if self.multiprocessing_distributed:
                # For multiprocessing distributed training, rank needs to be the
                # global rank among all the processes
                self.rank = self.rank * ngpus_per_node + gpu
            dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                    worldsize=self.worldsize, rank=self.rank)

        Log.log_info('=> Creating model...')
        model = self.model_fn(**self.model_args)

        if not torch.cuda.is_available():
            Log.log_info('using CPU, this will be slow...')
        elif self.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if self.gpu is not None:
                torch.cuda.set_device(self.gpu)
                model.cuda(self.gpu)
                if self.criterion_fn == "custom":
                    criterion = model.loss_fn
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                self.batchsize   = int(self.batchsize / ngpus_per_node)
                self.num_workers = int((self.num_workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
                Log.log_info('using fixed GPU {} with Distributed...'.format(self.gpu))
            else:
                model.cuda()
                if self.criterion_fn == "custom":
                    criterion = model.loss_fn
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
                Log.log_info('using all available GPUs with Distributed...')
        elif self.gpu is not None:
            Log.log_info('using fixed GPU {}'.format(self.gpu))
            torch.cuda.set_device(self.gpu)
            model = model.cuda(self.gpu)
            if self.criterion_fn == "custom":
                criterion = model.loss_fn
        else:
            Log.log_info('using all available GPUs...')
            if self.criterion_fn == "custom":
                criterion = model.loss_fn
            model = torch.nn.DataParallel(model).cuda()
        
        if self.criterion_fn != "custom":
            criterion = self.criterion_fn()
        optimizer = self.optimizer_fn(model.parameters(), **self.optimizer_args)
        scheduler = self.scheduler_fn(optimizer, **self.scheduler_args)

        self._load()

        if self.distributed:
            train_sampler = DistributedSampler(self.train_dataset)
            test_sampler  = DistributedSampler(self.test_dataset, shuffle=False, drop_last=True)
        else:   
            train_sampler = None
            test_sampler  = None

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batchsize, shuffle=(train_sampler is None),
            num_workers=self.num_workers, pin_memory=True, sampler=train_sampler)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batchsize, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, sampler=test_sampler)

        if not self.training:
            self.validate(self.test_loader, model, criterion)
            return

        for task in range(self.num_tasks):
            self._convert_task(task)
            Log.log('')
            Log.log('=> Task {} :'.format(task))
            for epoch in range(self.epoch, self.epochs):
                if self.distributed:
                    train_sampler.set_epoch(epoch)
                # train for one epoch
                try :
                    model.module.get_task(self._class_per_task[task])
                except Exception as e:
                    pass
                try :
                    model.get_task(self._class_per_task[task])
                except Exception as e:
                    pass
                self.train(self.train_loader, model, criterion, optimizer, epoch)
                Log.log('')
                # evaluate on validation set
                for test in range(task + 1):
                    self._convert_test(test)
                    Log.log('==> test for Task {} :'.format(test))
                    acc1 = self.validate(self.test_loader, model, criterion)
                scheduler.step()

        #setting up the distributed environment
        self.writer = SummaryWriter(self.save_path)

        return

    def train(self, train_loader, model, criterion, optimizer, epoch):
        self.log_interval = len(train_loader) // self.log_freqency
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        model.train()

        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if self.gpu is not None:
                images = images.cuda(self.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(self.gpu, non_blocking=True)
            with torch.cuda.amp.autocast(self.use_amp):
                # compute output
                output = model(images)
                loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if  i % self.log_interval == self.log_interval - 1 or i == len(train_loader) - 1:
                progress.display(i + 1)

    def validate(self, val_loader, model, criterion):
        def run_validate(loader, base_progress=0):
            with torch.no_grad():
                end = time.time()
                for i, (images, target) in enumerate(loader):
                    i = base_progress + i
                    if self.gpu is not None:
                        images = images.cuda(self.gpu, non_blocking=True)
                    if torch.cuda.is_available():
                        target = target.cuda(self.gpu, non_blocking=True)

                    # compute output
                    output = model(images)
                    loss = criterion(output, target)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if i % self.log_interval == self.log_interval - 1 or i == len(loader) - 1:
                        progress.display(i + 1)

        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        losses = AverageMeter('Loss', ':.4e', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
        progress = ProgressMeter(
            len(val_loader) + (self.distributed and (len(val_loader.sampler) * self.worldsize < len(val_loader.dataset))),
            [batch_time, losses, top1, top5],
            prefix='Test: ')

        # switch to evaluate mode
        model.eval()

        run_validate(val_loader)
        if self.distributed:
            top1.all_reduce()
            top5.all_reduce()

        if self.distributed and (len(val_loader.sampler) * self.worldsize < len(val_loader.dataset)):
            aux_val_dataset = Subset(val_loader.dataset,
                                    range(len(val_loader.sampler) * self.worldsize, len(val_loader.dataset)))
            aux_val_loader = DataLoader(
                aux_val_dataset, batch_size=self.batchsize, shuffle=False,
                num_workers=self.num_workers, pin_memory=True)
            run_validate(aux_val_loader, len(val_loader))

        progress.display_summary()

        return top1.avg

    def _dataset(self, _data : str) -> Dataset:
        if _data == 'CIFAR100':
            self.model_args["class_num"] = 100
            return CIFAR100
        elif _data == 'CIFAR10':
            self.model_args["class_num"] = 10
            return CIFAR10
        elif _data == 'MNIST':
            self.model_args["class_num"] = 10
            return MNIST
        elif _data == 'FashionMNIST':
            self.model_args["class_num"] = 10
            return FashionMNIST
        else:
            raise ValueError('Dataset {} not supported'.format(_data))

    def _save(self, **kwargs):
        r'''
        Save the model.
        '''
        torch.save({
            'epoch'                : self.epoch,
            'model_state_dict'     : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'scaler_state_dict'    : self.scaler.state_dict(),
            'scheduler_state_dict' : self.scheduler.state_dict(),
            'metrics'              : self._metrics,
            'step'                 : self._counts,
        }, os.path.join(self.save_dir, 'checkpoint_{}.pth'.format(self.epoch)))
        Log.log_info("Saved checkpoint to {}".format(os.path.join(self.save_dir, 'checkpoint_{}.pth'.format(self.epoch))))
        return
    def _load(self, *args, **kwargs):
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
            self.scheduler.load_state_dict   (load_dict['scheduler_state_dict'])
            self.epoch      = load_dict['epoch']
            self._metrics   = load_dict['metrics']
            self._counts    = load_dict['step']
        except:
            Log.log_info("Load failed. Start from Scratch.")
        Log.log_info("Loaded model from epoch {}".format(self.epoch))
        return

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        Log.log('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        Log.log(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
