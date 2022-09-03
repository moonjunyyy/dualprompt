import os
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from helper.log import Log
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from helper.metric import AverageMeter, ProgressMeter, Summary, accuracy

from utils.sampler import CILSampler

########################################################################################################################
# This is trainer with a DistributedDataParallel                                                                       #
# Based on the following tutorial:                                                                                     #
# https://github.com/pytorch/examples/blob/main/imagenet/main.py                                                       #
# And Deit by FaceBook                                                                                                 #
# https://github.com/facebookresearch/deit                                                                             #      
########################################################################################################################

class trainer():

    def __init__(self, args) -> None:

        self.arguments = args
        args.epoch          = 1
        transform = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor ()])
        args.dataset_train  = args.dataset(args.dataset_path, download=True, train=True,  transform=transform)
        args.dataset_val    = args.dataset(args.dataset_path, download=True, train=False, transform=transform)
        args.task_id        = 0
        args.test_id        = 0
        args.log_interval   = 0
        args.training    = True
        
        mp.set_start_method('fork')

        device = torch.device(args.device)
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.gpu = int(os.environ['LOCAL_RANK'])
            args.distributed = True
        elif 'SLURM_PROCID' in os.environ:
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
            args.distributed = True
        else: 
            args.rank = None
            args.world_size = None
            args.gpu = torch.cuda.device_count() - 1
            args.distributed = False

        if args.distributed:
            torch.cuda.set_device(args.gpu)
            args.dist_backend = 'nccl'
            print('| distributed init (rank {}): {}'.format(
                args.rank, args.dist_url), flush=True)
            dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                    world_size=args.world_size, rank=args.rank)
            dist.barrier()
            self._setup_for_distributed(args.rank == 0)
        
        if args.seed is not None:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            cudnn.deterministic = True
            Log.log_info('You have chosen to seed training. '
                         'This will turn on the CUDNN deterministic setting, '
                         'which can slow down your training considerably! '
                         'You may see unexpected behavior when restarting '
                         'from checkpoints.')
        self.device = torch.device(args.device)

        if args.task_governor is not None:
            Log.log_warning('Task governor is not implemented yet. Ignore the keyword and works CIL setting.')
            args.task_governor = None
        
        world_size  = self._get_world_size() if args.distributed else None
        global_rank = self._get_rank()       if args.distributed else None

        args.sampler_train = CILSampler(
            args.dataset_train, num_tasks=args.num_tasks, num_replicas=world_size, rank=global_rank, seed=args.seed, shuffle=True)
        args.sampler_val = CILSampler(
            args.dataset_val, num_tasks=args.num_tasks, num_replicas=world_size, rank=global_rank, seed=args.seed, shuffle=False)
        if args.distributed:
            if len(args.dataset_val) % world_size != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')

        args.data_loader_train = torch.utils.data.DataLoader(
            args.dataset_train, sampler=args.sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        args.data_loader_val = torch.utils.data.DataLoader(
            args.dataset_val, sampler=args.sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        Log.log_info(f"Creating model...")
        args.model = args.model(**args.model_args)
        args.model_without_ddp = args.model
        args.model.to(device)
        if args.distributed:
            args.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.gpu])
        args.optimizer = args.optimizer(args.model.parameters(), **args.optimizer_args)
        args.criterion = args.model_without_ddp.loss_fn if args.criterion == "custom" else args.criterion()
        args.scheduler = args.scheduler(args.optimizer, **args.scheduler_args)
        args.scaler    = torch.cuda.amp.GradScaler(enabled=args.use_amp)

        self._load()

    def __call__(args, **kwargs):
        return args.run(**kwargs)

    def _convert_train_task(self, task_idx):
        args = self.arguments
        args.sampler_train.set_task(task_idx)

    def _convert_test_task(self, task_idx):
        args = self.arguments
        args.sampler_test.set_task(task_idx)

    def run(self):
        ngpus_per_node    = torch.cuda.device_count()
        if self.arguments.distributed:
            # Since we have ngpus_per_node processes per node, the total worldsize
            # needs to be adjusted accordingly
            self.arguments.world_size = ngpus_per_node * self.arguments.world_size
            if self.arguments.world_size == 0:
                Log.log_exception('WORLD_SIZE is not set correctly. Please set it to the number of GPUs in your system.')
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # _worker process function
            mp.spawn(self._worker, nprocs=ngpus_per_node, args=(ngpus_per_node, self.arguments))
        else:
            # Simply call _worker function
            self._worker(self.arguments.gpu, self.arguments)
        return

    def _worker(self, gpu, args):
        args.ngpus_per_node = torch.cuda.device_count()
        if gpu is not None:
            Log.log('Using GPU:{} for training.'.format(args.gpu))

        if not args.training:
            self.validate(args.test_loader, args.model, args.criterion, args)
            return

        for task in range(args.num_tasks):
            self._convert_train_task(task)
            Log.log('')
            Log.log_info('Train Task {} :'.format(task))
            for epoch in range(args.epoch, args.epochs):
                args.sampler_train.set_epoch(epoch)
                args.sampler_val.set_epoch(epoch)
                # train for one epoch
                try :
                    args.model_without_ddp.set_task(args.dataset_train.get_task[task])
                except Exception as e: pass
                self.train(args.data_loader_train, args.model, args.criterion, args.optimizer, epoch, args)
                Log.log('')
                args.scheduler.step()

            # evaluate on validation set
            for test in range(task + 1):
                Log.log('==> test for Task {} :'.format(test))
                acc1 = self.validate(args.data_loader_val, args.model, args.criterion, args)
            
        torch.cuda.synchronize()
        print(Log._Log)
        return

    def train(self, train_loader, model, criterion, optimizer, epoch, args):
        args.log_interval = len(train_loader) // args.log_freqency
        batch_time = AverageMeter('Time', ':6.3f')
        data_time  = AverageMeter('Data', ':6.3f')
        losses     = AverageMeter('Loss', ':.4e')
        top1       = AverageMeter('Acc@1', ':6.2f')
        top5       = AverageMeter('Acc@5', ':6.2f')
        progress   = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        model.train()

        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)
            with torch.cuda.amp.autocast(args.use_amp):
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
            args.scaler.scale(loss).backward()
            args.scaler.step(optimizer)
            args.scaler.update()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if  i % args.log_interval == args.log_interval - 1 or i == len(train_loader) - 1:
                progress.display(i + 1)

    def validate(self, val_loader, model, criterion, args):
        def run_validate(loader, base_progress=0):
            with torch.no_grad():
                end = time.time()
                for i, (images, target) in enumerate(loader):
                    i = base_progress + i
                    if args.gpu is not None:
                        images = images.cuda(args.gpu, non_blocking=True)
                    if torch.cuda.is_available():
                        target = target.cuda(args.gpu, non_blocking=True)

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

                    if i % args.log_interval == args.log_interval - 1 or i == len(loader) - 1:
                        progress.display(i + 1)

        batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
        losses = AverageMeter('Loss', ':.4e', Summary.NONE)
        top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
        progress = ProgressMeter(
            len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.worldsize < len(val_loader.dataset))),
            [batch_time, losses, top1, top5],
            prefix='Test: ')

        # switch to evaluate mode
        model.eval()

        run_validate(val_loader)
        if args.distributed:
            top1.all_reduce()
            top5.all_reduce()

        if args.distributed and (len(val_loader.sampler) * args.worldsize < len(val_loader.dataset)):
            aux_val_dataset = Subset(val_loader.dataset,
                                    range(len(val_loader.sampler) * args.worldsize, len(val_loader.dataset)))
            aux_val_loader = DataLoader(
                aux_val_dataset, batch_size=args.batchsize, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)
            run_validate(aux_val_loader, len(val_loader))

        progress.display_summary()

        return top1.avg

    def _dataset(args, _data : str) -> Dataset:
        if _data == 'CIFAR100':
            args.model_args["class_num"] = 100
            return CIFAR100
        elif _data == 'CIFAR10':
            args.model_args["class_num"] = 10
            return CIFAR10
        elif _data == 'MNIST':
            args.model_args["class_num"] = 10
            return MNIST
        elif _data == 'FashionMNIST':
            args.model_args["class_num"] = 10
            return FashionMNIST
        else:
            raise ValueError('Dataset {} not supported'.format(_data))

    def _save(self):
        r'''
        Save the model.
        '''
        torch.save({
            'epoch'                : self.args.epoch,
            'model_state_dict'     : self.args.model_without_ddp.state_dict(),
            'optimizer_state_dict' : self.args.optimizer.state_dict(),
            'scaler_state_dict'    : self.args.scaler.state_dict(),
            'scheduler_state_dict' : self.args.scheduler.state_dict(),
        }, os.path.join(self.save_dir, 'checkpoint_{}.pth'.format(self.epoch)))
        Log.log_info("Saved checkpoint to {}".format(os.path.join(self.save_dir, 'checkpoint_{}.pth'.format(self.epoch))))
        return
    def _load(self, load_idx = -1):
        r'''
        Load the model.
        '''
        try:
            for e in range(self.args.epochs + 1):
                load_dict = torch.load(os.path.join(self.args.save_dir, 'checkpoint_{}.pth'.format(e)))
        except:
            pass
        try:
            self.args.model_without_ddp.load_state_dict(load_dict['model_state_dict'])
            self.args.optimizer.load_state_dict(load_dict['optimizer_state_dict'])
            self.args.scaler.load_state_dict   (load_dict['scaler_state_dict'])
            self.args.scheduler.load_state_dict(load_dict['scheduler_state_dict'])
            self.args.epoch    = load_dict['epoch']
            self.args._metrics = load_dict['metrics']
            self.args._counts  = load_dict['step']
        except:
            Log.log_info("Load failed. Start from Scratch.")
            return
        Log.log_info("Loaded model from epoch {}".format(self.args.epoch))
        return

    def _is_dist_avail_and_initialized(self):
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    def _get_world_size(self):
        if not self._is_dist_avail_and_initialized():
            return 1
        return dist.get_world_size()

    def _get_rank(self):
        if not self._is_dist_avail_and_initialized():
            return 0
        return dist.get_rank()

    def _is_main_process(self):
        return self._get_rank() == 0

    def _setup_for_distributed(is_master):
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

    def save_on_master(self, *args, **kwargs):
        if self._is_main_process():
            torch.save(*args, **kwargs)