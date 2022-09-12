import os
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from helper.metric import AverageMeter, ProgressMeter, Summary, accuracy
import builtins
from torch.utils.data import DataLoader, Subset

from utils.sampler import CILSampler

########################################################################################################################
# This is trainer with a DistributedDataParallel                                                                       #
# Based on the following tutorial:                                                                                     #
# https://github.com/pytorch/examples/blob/main/imagenet/main.py                                                       #
# And Deit by FaceBook                                                                                                 #
# https://github.com/facebookresearch/deit                                                                             #
########################################################################################################################

# TODO : Other Task Settings (TIL, Task Agnostic)
# TODO : Multi Node Distribution Code

class Imgtrainer():
    def __init__(self,
                 model, model_args,
                 criterion,
                 optimizer, optimizer_args,
                 scheduler, scheduler_args,
                 batch_size, step_size, epochs, log_frequency,
                 task_governor, num_tasks,
                 dataset, num_workers, dataset_path, save_path,
                 seed, device, pin_mem, use_amp, debug,
                 world_size, dist_url, dist_backend, rank, local_rank,
                 *args, **kwargs) -> None:
        
        self.model, self.model_args = model, model_args
        self.criterion = criterion
        self.optimizer, self.optimizer_args = optimizer, optimizer_args
        self.scheduler, self.scheduler_args = scheduler, scheduler_args

        self.epoch  = 0
        self.epochs = epochs
        self.batch_size    = batch_size
        self.step_size     = int(step_size // batch_size)
        self.log_frequency = log_frequency
        self.task_governor = task_governor
        
        self.training     = True
        self.lock         = mp.Lock()
        self.num_tasks    = num_tasks
        self.num_workers  = num_workers
        self.dataset_path = dataset_path
        self.save_path    = save_path
        self.pin_mem      = pin_mem
        self.use_amp      = use_amp
        self.seed         = seed
        self.device       = device
        self.debug        = debug

        self.rank         = rank
        self.dist_backend = dist_backend
        self.dist_url     = dist_url
        self.local_rank   = local_rank
        self.world_size   = world_size
        self.ngpus_per_nodes = torch.cuda.device_count()

        if "WORLD_SIZE" in os.environ:
            self.world_size  = int(os.environ["WORLD_SIZE"])
        else:
            self.world_size  = self.world_size * self.ngpus_per_nodes
        self.distributed     = self.world_size > 1

        # Transform needs to be diversed and be selected by user
        transform = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor ()])
        self.dataset_train   = dataset(dataset_path, download=True, train=True,  transform=transform)
        self.dataset_val     = dataset(dataset_path, download=True, train=False, transform=transform)

    def run(self):
        if self.distributed: 
            processes = []
            for n in range(self.ngpus_per_nodes):
                print(f"start process {n}...")
                p = mp.Process(target=self.main_worker, args=(n,))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            return
    
        else:
            self.main_worker(self.local_rank)

    def main_worker(self, gpu):
        self.gpu    = gpu % self.ngpus_per_nodes
        self.device = torch.device(self.gpu)
        if self.distributed:
            #os.environ['MASTER_PORT'] = str(int(os.environ['MASTER_PORT']) + self.rank * self.ngpus_per_nodes + gpu)
            self.local_rank = gpu
            if 'SLURM_PROCID' in os.environ.keys():
                self.rank = int(os.environ['SLURM_PROCID']) * self.ngpus_per_nodes + gpu
                print(f"| Init Process group {os.environ['SLURM_PROCID']} : {self.local_rank}")
            else :
                self.rank = gpu
                print(f"| Init Process group 0 : {self.local_rank}")

            if 'MASTER_ADDR' not in os.environ.keys():
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '12701'
            dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                    world_size=self.world_size, rank=self.rank)
            self.setup_for_distributed(self.is_main_process())
        else:
            pass
        if self.seed is not None:
            seed = self.seed + self.rank
            random.seed(seed)
            torch.manual_seed(seed)
            cudnn.deterministic = True
            print('You have chosen to seed training. '
                'This will turn on the CUDNN deterministic setting, '
                'which can slow down your training considerably! '
                'You may see unexpected behavior when restarting '
                'from checkpoints.')
        cudnn.benchmark = True

        _r = dist.get_rank() if self.distributed else None # means that it is not distributed
        _w = dist.get_world_size() if self.distributed else None # means that it is not distributed
        sampler_train = CILSampler(self.dataset_train, self.num_tasks, _w, _r, shuffle=True, seed=self.seed)
        sampler_val   = CILSampler(self.dataset_val  , self.num_tasks, _w, _r, shuffle=False, seed=self.seed)
        self.batch_size = int(self.batch_size // self.world_size)
        
        print("Building model...")
        model = self.model(**self.model_args)
        model.to(self.device)
        model_without_ddp = model
        if self.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model)
            model._set_static_graph()
            model_without_ddp = model.module
        criterion = model_without_ddp.loss_fn if self.criterion == 'custom' else self.criterion()
        optimizer = self.optimizer(model.parameters(), **self.optimizer_args)
        scheduler = self.scheduler(optimizer, **self.scheduler_args)

        n_params = sum(p.numel() for p in model_without_ddp.parameters())
        print(f"Total Parameters :\t{n_params}")
        n_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        print(f"Learnable Parameters :\t{n_params}")
        print("")

        if not self.training:
                self.validate(loader_val, model, criterion)
                return
        for self.task in range(self.num_tasks):
            loader_train = self.set_task(self.dataset_train, sampler_train, self.task)
            print("Selection : ",(model_without_ddp._convert_train_task(sampler_train.get_task()).to(torch.int) - 1).tolist())
            print(f"Training for task {self.task} : {sampler_train.get_task().tolist()}")

            for self.epoch in range(self.epochs):
                sampler_train.set_epoch(self.epoch)
                self.train(loader_train, model, criterion, optimizer)
                print('')
                scheduler.step()

            for self.test in range(self.task + 1):
                loader_val = self.set_task(self.dataset_val, sampler_val, self.test) 
                self.validate(loader_val, model, criterion)

            self.epoch = 0
            optimizer = self.optimizer(model.parameters(), **self.optimizer_args)
            print('')
        print("Selection : ",(model_without_ddp._convert_train_task(sampler_train.get_task())-1).tolist())
    
    def set_task(self, dataset, sampler, task):
        sampler.set_task(task)
        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            sampler=sampler,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_mem)  
        return loader

    def train(self, loader, model, criterion, optimizer):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time  = AverageMeter('Data', ':6.3f')
        losses     = AverageMeter('Loss', ':.4e')
        top1       = AverageMeter('Acc@1', ':6.2f')
        top5       = AverageMeter('Acc@5', ':6.2f')
        progress   = ProgressMeter(
            len(loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(self.epoch + 1))
        # switch to train mode
        model.train()
        log_interval = int(len(loader) // self.log_frequency)
        end = time.time()
        for i, (images, target) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)
            images, target = images.to(self.device), target.to(self.device)
            # compute output
            output = model(images)
            loss = criterion(output, target)
            torch.cuda.synchronize()
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            torch.cuda.synchronize()
            # compute gradient and do SGD step
            loss.backward()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.step_size == self.step_size - 1 or i == len(loader) - 1:
                optimizer.step()
                optimizer.zero_grad()
            if i % log_interval == log_interval - 1 or i == len(loader) - 1:
                progress.display(i + 1)
        progress.write_summary(epoch = self.epoch, save_path = self.save_path, prefix='Train/')
                
    def validate(self, loader, model, criterion):

        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        losses = AverageMeter('Loss', ':.4e', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

        progress = ProgressMeter(
            len(loader),
            [batch_time, losses, top1, top5],
            prefix='Test: ')

        # switch to evaluate mode
        model.eval()
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                images, target = images.to(self.device), target.to(self.device)
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
        torch.cuda.synchronize()
        if self.distributed:
            top1.all_reduce()
            top5.all_reduce()
        progress.display_summary()
        progress.write_summary(epoch = self.test, save_path = self.save_path, prefix='Test/task{}/'.format(self.task))

        return top1.avg
    def save_on_master(self):
        if self.is_main_process():
            self.save(self)

    def is_dist_avail_and_initialized(self):
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    def get_world_size(self):
        if not self.is_dist_avail_and_initialized():
            return 1
        return dist.get_world_size()

    def get_rank(self):
        if not self.is_dist_avail_and_initialized():
            return 0
        return dist.get_rank()

    def is_main_process(self):
        return self.get_rank() == 0

    def setup_for_distributed(self, is_master):
        """
        This function disables printing when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)
        __builtin__.print = print

    def save(self):
            r'''
            Save the model.
            '''
            torch.save({
                'epoch'                : self.epoch,
                'model_state_dict'     : self.model.state_dict(),
                'optimizer_state_dict' : self.optimizer.state_dict(),
                'scheduler_state_dict' : self.scheduler.state_dict(),
            }, os.path.join(self.save_path, 'checkpoint_{}.pth'.format(self.epoch)))
            print("Saved checkpoint to {}".format(os.path.join(self.save_path, 'checkpoint_{}.pth'.format(self.epoch))))
            return
            
    def load(self, load_idx = -1):
        r'''
        Load the model.
        '''
        try:
            for e in range(1, self.epochs + 1 if load_idx == -1 else load_idx):
                load_dict = torch.load(os.path.join(self.save_path, 'checkpoint_{}.pth'.format(e)))
        except:
            pass
        try:
            self.model.load_state_dict(load_dict['model_state_dict'])
            self.optimizer.load_state_dict(load_dict['optimizer_state_dict'])
            self.scheduler.load_state_dict(load_dict['scheduler_state_dict'])
            self.epoch = load_dict['epoch']
        except:
            print("Load failed. Start from Scratch.")
            return
        print("Loaded model from epoch {}".format(self.epoch))
        return
