from ast import arg
import copy
import os
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from helper.log import Log
from helper.metric import AverageMeter, ProgressMeter, Summary, accuracy
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST

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

def _save(args):
        r'''
        Save the model.
        '''
        torch.save({
            'epoch'                : args.epoch,
            'model_state_dict'     : args.model.state_dict(),
            'optimizer_state_dict' : args.optimizer.state_dict(),
            'scaler_state_dict'    : args.scaler.state_dict(),
            'scheduler_state_dict' : args.scheduler.state_dict(),
        }, os.path.join(args.save_dir, 'checkpoint_{}.pth'.format(args.epoch)))
        print("Saved checkpoint to {}".format(os.path.join(args.save_dir, 'checkpoint_{}.pth'.format(args.epoch))))
        return
        
def _load(args, load_idx = -1):
    r'''
    Load the model.
    '''
    try:
        for e in range(1, args.epochs + 1):
            load_dict = torch.load(os.path.join(args.save_dir, 'checkpoint_{}.pth'.format(e)))
    except:
        pass
    try:
        args.model.load_state_dict(load_dict['model_state_dict'])
        args.optimizer.load_state_dict(load_dict['optimizer_state_dict'])
        args.scaler.load_state_dict   (load_dict['scaler_state_dict'])
        args.scheduler.load_state_dict(load_dict['scheduler_state_dict'])
        epoch    = load_dict['epoch']
    except:
        print("Load failed. Start from Scratch.")
        return
    print("Loaded model from epoch {}".format(epoch))
    return args

def save_on_master(self, *args, **kwargs):
    if self.is_main_process():
        torch.save(*args, **kwargs)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def setup_for_distributed(is_master):
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

def worker(gpu, ngpus_per_node, args):

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    args.rank = gpu
    args.gpu = args.rank % ngpus_per_node

    if gpu is not None:
        print('Using GPU:{} for training.'.format(args.gpu))

    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(args.rank, args.gpu))
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            rank=args.rank, world_size = args.world_size)
    
    if args.seed is not None:
        seed = args.seed + get_rank()
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        print('You have chosen to seed training. '
              'This will turn on the CUDNN deterministic setting, '
              'which can slow down your training considerably! '
              'You may see unexpected behavior when restarting '
              'from checkpoints.')
        cudnn.benchmark = True

    torch.cuda.set_device(args.gpu)
    setup_for_distributed(is_main_process())

    world_size  = get_world_size() if args.distributed else None
    global_rank = get_rank()       if args.distributed else None

    args.sampler_train = CILSampler(
        args.dataset_train, num_tasks=args.num_tasks, num_replicas=args.world_size, rank=global_rank, seed=args.seed, shuffle=True)
    args.sampler_val   = CILSampler(
        args.dataset_val,   num_tasks=args.num_tasks, num_replicas=args.world_size, rank=global_rank, seed=args.seed, shuffle=False)
    if args.distributed:
        if len(args.dataset_val) % args.world_size != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
    args.batch_size = int(args.batch_size / args.world_size)

    args.data_loader_train = torch.utils.data.DataLoader(
        args.dataset_train, sampler=args.sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    args.data_loader_val   = torch.utils.data.DataLoader(
        args.dataset_val, sampler=args.sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    print(f"Creating model...")
    args.model = args.model(**args.model_args)
    args.model_without_ddp = args.model
    args.model.cuda(args.gpu)
    if args.distributed:
        args.model = torch.nn.parallel.DistributedDataParallel(args.model)
        args.model._set_static_graph()
    args.optimizer = args.optimizer(args.model.parameters(), **args.optimizer_args)
    args.criterion = args.model_without_ddp.loss_fn if args.criterion == "custom" else args.criterion()
    args.scheduler = args.scheduler(args.optimizer, **args.scheduler_args)
    args.scaler    = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    args.epoch = 1
    _load(args)

    dist.barrier()
    if not args.training:
        args.log_interval = len(args.data_loader_val) // args.log_freqency
        validate(args)
        return
    for task in range(args.num_tasks):
        args.log_interval = len(args.data_loader_train) // args.log_freqency
        args.sampler_train.set_task(task)
        print('')
        print('Train Task {} :'.format(task))
        for epoch in range(args.epoch, args.epochs + 1):
            args.sampler_train.set_epoch(epoch)
            # train for one epoch
            try :
                args.model_without_ddp.set_task(args.dataset_train.get_task[task])
            except Exception as e: pass
            train(args)
            print('')
            args.scheduler.step()
        # evaluate on validation set
        for test in range(task + 1):
            print('==> test for Task {} :'.format(test))
            args.sampler_val.set_task(test)
            acc1 = validate(args)
    return

def train(args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time  = AverageMeter('Data', ':6.3f')
    losses     = AverageMeter('Loss', ':.4e')
    top1       = AverageMeter('Acc@1', ':6.2f')
    top5       = AverageMeter('Acc@5', ':6.2f')
    progress   = ProgressMeter(
        len(args.data_loader_train),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(args.epoch))

    # switch to train mode
    args.model.train()

    end = time.time()
    for i, (images, target) in enumerate(args.data_loader_train):
        # measure data loading timeorprint
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)
        with torch.cuda.amp.autocast(args.use_amp):
            # compute output
            output = args.model(images)
            loss = args.criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        args.optimizer.zero_grad()
        args.scaler.scale(loss).backward()
        args.scaler.step(args.optimizer)
        args.scaler.update()

        torch.cuda.synchronize()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if  i % args.log_interval == 0 or i == len(args.data_loader_train) - 1:
            progress.display(i + 1)

def validate(args):
    def run_validate(loader, base_progress=0):
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                i = base_progress + i
                if args.gpu is not None:
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    target = target.cuda(args.gpu, non_blocking=True)

                with torch.cuda.amp.autocast(args.use_amp):
                    # compute output
                    output = args.model(images)
                    loss = args.criterion(output, target)
                
                torch.cuda.synchronize()
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

              #  if i % args.log_interval == args.log_interval - 1 or i == len(loader) - 1:
             #       progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(args.data_loader_val) + (args.distributed and (len(args.data_loader_val.sampler) * args.world_size < len(args.data_loader_val.dataset))),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    args.model.eval()

    run_validate(args.data_loader_val)
    if args.distributed:
        top1.all_reduce()
        top5.all_reduce()

    if args.distributed and (len(args.data_loader_val.sampler) * args.world_size < len(args.data_loader_val.dataset)):
        aux_val_dataset = Subset(args.data_loader_val.dataset,
                                range(len(args.data_loader_val.sampler) * args.world_size, len(args.data_loader_val.dataset)))
        aux_val_loader = DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
        run_validate(aux_val_loader, len(args.data_loader_val))

    progress.display_summary()

    return top1.avg

def image_trainer(args):
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor ()])
    args.dataset_train  = args.dataset(args.dataset_path, download=True, train=True,  transform=transform)
    args.dataset_val    = args.dataset(args.dataset_path, download=True, train=False, transform=transform)
    args.epoch          = 1
    args.task_id        = 0
    args.test_id        = 0
    args.log_interval   = 0
    args.training       = True
    args.world_size     = args.num_nodes * torch.cuda.device_count()
    args.distributed    = args.world_size > 1
    if args.task_governor is not None:
        print('Task governor is not implemented yet. Ignore the keyword and works CIL setting.')
        args.task_governor = None

    mp.set_start_method('spawn')
    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        processes = []
        for rank in range(args.world_size):
            print("start rank {} ...".format(rank))
            p = mp.Process(target = worker, args = (rank, ngpus_per_node, args))
            p.start()
            processes.append(p)
        #mp.spawn(fn = worker, args = (ngpus_per_node, self.args), nprocs = self.args.world_size)
        for p in processes:
            p.join()
    else:
        worker(ngpus_per_node, ngpus_per_node, args)
    return