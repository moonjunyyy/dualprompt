import os
import random
import time

import pprint
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import wandb
from models.CPP import CPP
from utils.args import (get_criterion, get_dataset, get_model, get_optimizer,
                          get_scheduler)
from utils.metric import AverageMeter, ProgressMeter, Summary, accuracy
from models.GausskeyL2P import GausskeyL2P
import json
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet
from datasets import multiDatasets
from sklearn.manifold import TSNE

from utils.sampler import CILSampler, multiDatasetSampler

########################################################################################################################
# This is trainer with a DistributedDataParallel                                                                       #
# Based on the following tutorial:                                                                                     #
# https://github.com/pytorch/examples/blob/main/imagenet/main.py                                                       #
# And Deit by FaceBook                                                                                                 #
# https://github.com/facebookresearch/deit                                                                             #
########################################################################################################################

# TODO : Other Task Settings (TIL, Task Agnostic)
class Imgtrainer():
    
    def __init__(self,
                 project, name, entity, sweep,
                 model, model_args,
                 criterion,
                 optimizer, optimizer_args,
                 scheduler, scheduler_args,
                 batch_size, step_size, epochs, log_frequency, grad_clip,
                 task_governor, num_tasks,
                 dataset, num_workers, dataset_path, save_path, eval,
                 seed, device, pin_mem, use_amp, use_tf, debug,
                 world_size, dist_url, dist_backend, rank, local_rank,
                 *args) -> None:

        self.project = project
        self.entity  = entity
        self.name    = name
        self.sweep   = sweep

        # Model and Training Settings
        self.model_base, self.model_args = get_model(model), model_args
        self.criterion = get_criterion(criterion)
        self.optimizer_base, self.optimizer_args = get_optimizer(optimizer), optimizer_args
        self.scheduler_base, self.scheduler_args = get_scheduler(scheduler), scheduler_args

        self.epoch  = 0
        self.epochs = epochs
        self.batch_size    = batch_size
        self.step_size     = int(step_size // batch_size)
        self.log_frequency = log_frequency
        self.task_governor = task_governor # CIL or Single
        self.grad_clip     = grad_clip
        
        # Dataset Settings
        self.num_tasks    = num_tasks
        self.num_workers  = num_workers
        self.dataset      = get_dataset(dataset)
        self.dataset_path = dataset_path
        self.pin_mem      = pin_mem

        # Options
        self.training     = not eval
        self.save_path    = save_path
        self.use_amp      = use_amp
        self.scaler       = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.seed         = seed
        self.device       = device
        self.debug        = debug
        self.use_tf       = use_tf

        # Distributed Settings
        self.rank         = rank
        self.dist_backend = dist_backend
        self.dist_url     = dist_url
        self.local_rank   = local_rank
        self.world_size   = world_size
        self.ngpus_per_nodes = torch.cuda.device_count()

        if "WORLD_SIZE" in os.environ:
            self.world_size  = int(os.environ["WORLD_SIZE"]) * self.ngpus_per_nodes
        else:
            self.world_size  = self.world_size * self.ngpus_per_nodes
        self.distributed     = self.world_size > 1
        
        #TODO : Transform needs to be diversed and be selected by user
        self.train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor ()])
        self.var_transform   = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor ()])

        if self.dataset==ImageNet:
            #TODO : Transform needs to be diversed and be selected by user
            self.train_transform = transforms.Compose([transforms.AutoAugment(), transforms.Resize((224, 224)), transforms.ToTensor ()])
            self.var_transform   = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor ()])
            self.dataset_train   = self.dataset(self.dataset_path, split = 'train',  transform=self.train_transform)
            self.dataset_val     = self.dataset(self.dataset_path, split = 'val',    transform=self.var_transform)
        else:
            if len(self.dataset) == 1:
                self.dataset_train   = self.dataset(self.dataset_path, download=True, train=True,  transform=self.train_transform)
                self.dataset_val     = self.dataset(self.dataset_path, download=True, train=False, transform=self.var_transform)
            else:
                self.dataset_train   = multiDatasets(self.dataset, self.dataset_path, download=True, train=True,  transform=self.train_transform, task=0)
                self.dataset_val     = multiDatasets(self.dataset, self.dataset_path, download=True, train=False, transform=self.var_transform,   task=0)

    def run(self):
        if self.sweep:
            sweep_config = json.loads("sweep.json")
            sweep_id = wandb.sweep(sweep_config, project=self.project, entity=self.entity)
            wandb.agent(sweep_id, function=self.process_devider, project=self.project, entity=self.entity, count=100)
        else:
            self.process_devider()

    def process_devider(self):
        wandb.init()
        for key, value in wandb.config.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    getattr(self, key)[k] = v
            else:
                setattr(self, key, value)
        pprint.pprint(self.__dict__)
        wandb.finish()
        if self.ngpus_per_nodes > 1: 
            # mp.spawn(self.main_worker, (), self.ngpus_per_nodes, True)
            processes = []
            for i in range(0, self.ngpus_per_nodes):
                p = mp.Process(target=self.main_worker, args=(i,), )
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
        else:
            self.main_worker(self.local_rank)

    def main_worker(self, gpu) -> None:
        
        self.gpu    = gpu % self.ngpus_per_nodes
        self.device = torch.device(self.gpu)
        if self.distributed:
            self.local_rank = self.gpu
            if 'SLURM_PROCID' in os.environ.keys():
                self.rank = int(os.environ['SLURM_PROCID']) * self.ngpus_per_nodes + self.gpu
                print(f"| Init Process group {os.environ['SLURM_PROCID']} : {self.local_rank}")
            else :
                self.rank = self.gpu
                print(f"| Init Process group 0 : {self.local_rank}")
            if 'MASTER_ADDR' not in os.environ.keys():
                os.environ['MASTER_ADDR'] = '127.0.0.1'
                os.environ['MASTER_PORT'] = '12701'
            torch.cuda.set_device(self.gpu)
            time.sleep(self.rank * 0.1) # prevent port collision
            dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                    world_size=self.world_size, rank=self.rank)
            self.setup_for_distributed(self.is_main_process())
        else:
            pass

        if self.is_main_process():
            wandb.init(project=self.project, entity=self.entity, group=self.name, name=self.name)

        if self.seed is not None:
            seed = self.seed
            random.seed(seed)
            torch.manual_seed(seed)
            cudnn.deterministic = True
            print('You have chosen to seed training. '
                'This will turn on the CUDNN deterministic setting, '
                'which can slow down your training considerably! '
                'You may see unexpected behavior when restarting '
                'from checkpoints.')
        cudnn.benchmark = True

        if self.training:
            if self.task_governor == "CIL":
                if issubclass(self.model_base, CPP):
                    self.CPP_Train()
                    return
                self.CIL_Train()
            else:
                self.Single_Task_Train()
        else:
            if self.task_governor == "CIL":
                if issubclass(self.model_base, CPP):
                    self.CPP_Eval()
                    return
                self.CIL_Eval()
                
            else:
                self.Single_Task_Eval()

        if self.is_main_process():
            wandb.finish()

    def setup_dataset_for_distributed(self):
        _r = dist.get_rank() if self.distributed else None       # means that it is not distributed
        _w = dist.get_world_size() if self.distributed else None # means that it is not distributed

        if len(self.dataset)  == 1:
            self.sampler_train = CILSampler(self.dataset_train, self.num_tasks, _w, _r, shuffle=True,  shuffle_class=self.random_class, seed=self.seed)
            self.sampler_val   = CILSampler(self.dataset_val  , self.num_tasks, _w, _r, shuffle=False, shuffle_class=self.random_class, seed=self.seed)
        else:
            self.sampler_train = multiDatasetSampler(self.dataset_train, self.num_tasks, _w, _r, shuffle=True,  shuffle_class=self.random_class, seed=self.seed)
            self.sampler_val   = multiDatasetSampler(self.dataset_val  , self.num_tasks, _w, _r, shuffle=False, shuffle_class=self.random_class, seed=self.seed)
        
        self.batch_size = int(self.batch_size // self.world_size)

        print("Building model...")
        self.model = self.model_base(**self.model_args)
        self.model.to(self.device)
        # wandb.watch(self.model)
        self.model_without_ddp = self.model
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)
            self.model._set_static_graph()
            self.model_without_ddp = self.model.module
        self.criterion = self.model_without_ddp.loss_fn if self.criterion == 'custom' else self.criterion()
        self.optimizer = self.optimizer_base(self.model.parameters(), **self.optimizer_args)
        self.scheduler = self.scheduler_base(self.optimizer, **self.scheduler_args)

        self.load(self.model_without_ddp, self.optimizer, self.scheduler, self.epoch)

        n_params = sum(p.numel() for p in self.model_without_ddp.parameters())
        print(f"Total Parameters :\t{n_params}")
        n_params = sum(p.numel() for p in self.model_without_ddp.parameters() if p.requires_grad)
        print(f"Learnable Parameters :\t{n_params}")
        print("")

    def CPP_Train(self):
        self.setup_dataset_for_distributed()
        accuracy_matrix = torch.zeros((self.num_tasks, self.num_tasks))

        for self.task in range(self.num_tasks):     
            loader_train = self.set_task(self.dataset_train, self.sampler_train, self.task)
            self.model_without_ddp.convert_train_task(self.sampler_train.get_task())
            print(f"Training for task {self.task} : {self.sampler_train.get_task().tolist()}")
            
            self.model_without_ddp.pre_forward(loader_train)
            for self.epoch in range(self.epochs):
                self.sampler_train.set_epoch(self.epoch)
                self.train(loader_train, self.model, self.criterion, self.optimizer)
                self.scheduler.step()
                print('')
            self.model_without_ddp.post_forward(loader_train)

            print("Selection : ",(self.model_without_ddp.get_count().to(torch.int)).tolist(), end='\n\n')
            for self.test in range(self.task + 1):
                loader_val = self.set_task(self.dataset_val, self.sampler_val, self.test) 
                accuracy_matrix[self.task, self.test] = self.validate(loader_val, self.model, self.criterion)
                print("Selection : ",(self.model_without_ddp.get_count().to(torch.int)).tolist())
            self.epoch = 0
            self.optimizer = self.optimizer_base(self.model.parameters(), **self.optimizer_args)
            print('')

        print(f"Accuracy Matrix :\n{accuracy_matrix.numpy()} \n Average Accuracy : {accuracy_matrix[-1,:].mean().item()}")
        forgetting = accuracy_matrix.max(dim=0)[0] - accuracy_matrix[-1, :]
        print(f"Forgetting : {forgetting.numpy()} \n Average Forgetting : {forgetting.mean().item()}")
        if self.is_main_process():
            wandb.log({'Average_Accuracy' : accuracy_matrix[-1,:].mean().item(),'Average_Forgetting' : forgetting.mean().item()})
        self.save(self.model_without_ddp, self.optimizer, self.scheduler, self.epoch)

    def CIL_Train(self):
        self.setup_dataset_for_distributed()
        ViT_Features = [torch.empty((0, self.model_without_ddp.backbone.num_features), device=self.device) for i in range(self.num_tasks)]
        accuracy_matrix = torch.zeros((self.num_tasks, self.num_tasks))
        for self.task in range(self.num_tasks):
            loader_train = self.set_task(self.dataset_train, self.sampler_train, self.task)
            self.model_without_ddp.convert_train_task(self.sampler_train.get_task())
            print(f"Training for task {self.task} : {self.sampler_train.get_task().tolist()}")
            # Features for Tsne plot 
            for i, (data, target) in enumerate(loader_train):
                data = data.to(self.device)
                x = self.model_without_ddp.backbone.patch_embed(data)

                B, N, D = x.size()
                cls_token = self.model_without_ddp.backbone.cls_token.expand(B, -1, -1)
                t = torch.cat((cls_token, x), dim=1)
                x = self.model_without_ddp.backbone.pos_drop(t + self.model_without_ddp.backbone.pos_embed)
                q = self.model_without_ddp.backbone.blocks(x)
                q = self.model_without_ddp.backbone.norm(q)[:, 0].clone()
                ViT_Features[self.task] = torch.concat((ViT_Features[self.task], q))

            loader_train = self.set_task(self.dataset_train, self.sampler_train, self.task)
            for self.epoch in range(self.epochs):
                self.sampler_train.set_epoch(self.epoch)
                self.train(loader_train, self.model, self.criterion, self.optimizer)
                self.scheduler.step()
                print('')
            print("Selection : ",(self.model_without_ddp.get_count().to(torch.int)).tolist(), end='\n\n')
            for self.test in range(self.task + 1):
                loader_val = self.set_task(self.dataset_val, self.sampler_val, self.test) 
                accuracy_matrix[self.task, self.test] = self.validate(loader_val, self.model, self.criterion)
                print("Selection : ",(self.model_without_ddp.get_count().to(torch.int)).tolist())
            self.epoch = 0
            self.optimizer = self.optimizer_base(self.model.parameters(), **self.optimizer_args)
            print('')

        print(f"Accuracy Matrix :\n{accuracy_matrix.numpy()} \n Average Accuracy : {accuracy_matrix[-1,:].mean().item()}")
        forgetting = accuracy_matrix.max(dim=0)[0] - accuracy_matrix[-1, :]
        print(f"Forgetting : {forgetting.numpy()} \n Average Forgetting : {forgetting.mean().item()}")
        if self.is_main_process():
            wandb.log({'Average_Accuracy' : accuracy_matrix[-1,:].mean().item(),'Average_Forgetting' : forgetting.mean().item()})
        self.save(self.model_without_ddp, self.optimizer, self.scheduler, self.epoch)

        if self.is_main_process():
            if self.model_base == GausskeyL2P:
                N, D = ViT_Features[0].shape
                vec = torch.empty((0, D), device=ViT_Features[0].device)
                N = int(5000/self.num_tasks)     # Too much vectors make OOM Problem
                for n, f in enumerate(ViT_Features):
                    vec = torch.concat((vec, f[:N]), dim = 0)
                P, D = self.model_without_ddp.mean.shape
                vec = torch.concat((vec, self.model_without_ddp.mean), dim = 0)

                pd.DataFrame(vec.cpu().detach().numpy()).to_csv(f"{self.save_path}RAW_ViT_Features.csv")
                vec = TSNE().fit_transform((vec.cpu().detach().numpy()))
                pd.DataFrame(vec).to_csv(f"{self.save_path}ViT_Features.csv")
                for n in range(len(ViT_Features)):
                    plt.scatter(vec[N * n : N * (n + 1),0], vec[N * n : N * (n + 1),1], s=1)
                plt.scatter(vec[-self.model_without_ddp.pool_size:, 0], vec[-self.model_without_ddp.pool_size:, 1], s=15, marker='+', color='black')
                plt.axis()
                plt.savefig(f"{self.save_path}ViT_Features.png")
                plt.clf()

                P, L, D = self.model_without_ddp.prompt.shape
                vec = TSNE().fit_transform(self.model_without_ddp.prompt.reshape(-1, D).cpu().detach().numpy())
                pd.DataFrame(vec).to_csv(f"{self.save_path}prompts.csv")
                for p in range(P):
                    plt.scatter(vec[p*L : (p+1)*L, 0], vec[p*L : (p+1)*L, 1], s=1)
                plt.axis()
                plt.savefig(f"{self.save_path}prompts.png")
                plt.clf()
            else:
                N, D = ViT_Features[0].shape
                vec = torch.empty((0, D), device=ViT_Features[0].device)
                N = int(5000/self.num_tasks)     # Too much vectors make OOM Problem
                for n, f in enumerate(ViT_Features):
                    vec = torch.concat((vec, f[:N]), dim = 0)
                pd.DataFrame(vec.cpu().detach().numpy()).to_csv(f"{self.save_path}RAW_ViT_Features.csv")
                P, D = self.model_without_ddp.prompt.key.shape
                vec = torch.concat((vec, self.model_without_ddp.prompt.key), dim = 0)
                vec = TSNE().fit_transform((vec.cpu().detach().numpy()))
                pd.DataFrame(vec).to_csv(f"{self.save_path}ViT_Features.csv")
                for n in range(len(ViT_Features)):
                    plt.scatter(vec[N * n : N * (n + 1),0], vec[N * n : N * (n + 1),1], s=1)
                plt.scatter(vec[-self.model_without_ddp.prompt.pool_size:, 0], vec[-self.model_without_ddp.prompt.pool_size:, 1], s=15, marker='+', color='black')
                plt.axis()
                plt.savefig(f"{self.save_path}ViT_Features.png")
                plt.clf()

                P, L, D = self.model_without_ddp.prompt.prompts.shape
                vec = TSNE().fit_transform(self.model_without_ddp.prompt.prompts.reshape(-1, D).cpu().detach().numpy())
                pd.DataFrame(vec).to_csv(f"{self.save_path}prompts.csv")
                for p in range(P):
                    plt.scatter(vec[p*L : (p+1)*L, 0], vec[p*L : (p+1)*L, 1], s=1)
                plt.axis()
                plt.savefig(f"{self.save_path}prompts.png")
                plt.clf()
        return

    def Single_Task_Train(self):
        self.setup_dataset_for_distributed()

        loader_train = self.set_task(self.dataset_train, self.sampler_train, 0)
        loader_val   = self.set_task(self.dataset_val,   self.sampler_val,   0) 
        self.task = 0
        self.test = 0
        if not self.training:
            self.validate(loader_val, self.model, self.criterion)
            return

        for self.epoch in range(self.epochs):
            self.sampler_train.set_epoch(self.epoch)
            self.train(loader_train, self.model, self.criterion, self.optimizer)
            print('')
            self.scheduler.step()
            self.validate(loader_val, self.model, self.criterion)
            self.save(self.model_without_ddp, self.optimizer, self.scheduler, self.epoch)
        print('')
        return

    def set_task(self, dataset, sampler, task):
        sampler.set_task(task)
        loader = DataLoader(dataset,
                            batch_size =self.batch_size,
                            sampler    =sampler,
                            num_workers=self.num_workers,
                            pin_memory =self.pin_mem)  
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
        self.model.train()
        log_interval = int(len(loader) // self.log_frequency)
        end = time.time()
        for i, (images, target) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)
            images, target = images.to(self.device), target.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # compute output
                output = model(images)
                loss = criterion(output, target)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
                # compute gradient and do SGD step
                self.scaler.scale(loss).backward()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % self.step_size == self.step_size - 1 or i == len(loader) - 1:
                if self.grad_clip > 0.0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(optimizer)
                optimizer.zero_grad()
                self.scaler.update()
            if i % log_interval == log_interval - 1 or i == len(loader) - 1:
                progress.display(i + 1)
        if self.use_tf:
            progress.write_summary(epoch = self.task * self. epochs + self.epoch, save_path = self.save_path, prefix='Train/')
            
        if self.is_main_process():
            wandb.log({'Train/Loss': losses.avg, 'Train/Acc@1': top1.avg, 'Train/Acc@5': top5.avg}, step = self.task * self.epochs + self.epoch)
                
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
        self.model.eval()
        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    images, target = images.to(self.device), target.to(self.device)
                    # compute output
                    output = self.model(images)
                    loss = self.criterion(output, target)
                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))
                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()
        if self.distributed:
            dist.barrier()
            top1.all_reduce()
            top5.all_reduce()
        progress.display_summary()
        if self.use_tf:
            progress.write_summary(epoch = self.task, save_path = self.save_path, prefix='Test/task{}/'.format(self.test))
            
        if self.is_main_process():
            wandb.log({'Test/Loss': losses.avg, 'Test/Acc@1': top1.avg, 'Test/Acc@5': top5.avg}, step = self.task)
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

    def save(self, model, optimizer, scheduler, epoch):
            r'''
            Save the self.model.
            '''
            torch.save({
                'epoch'                : epoch,
                'model_state_dict'     : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'scheduler_state_dict' : scheduler.state_dict(),
            }, os.path.join(self.save_path, 'checkpoint_{}.pth'.format(self.epoch)))
            print("Saved checkpoint to {}".format(os.path.join(self.save_path, 'checkpoint_{}.pth'.format(self.epoch))))
            return True
            
    def load(self, model, optimizer, scheduler, epoch, load_idx = -1):
        r'''
        Load the self.model.
        '''
        try:
            for e in range(1, self.epochs + 1 if load_idx == -1 else load_idx):
                load_dict = torch.load(os.path.join(self.save_path, 'checkpoint_{}.pth'.format(e)))
        except:
            pass
        try:
            self.model.load_state_dict(load_dict['model_state_dict'])
            optimizer.load_state_dict(load_dict['optimizer_state_dict'])
            scheduler.load_state_dict(load_dict['scheduler_state_dict'])
            epoch = load_dict['epoch']
        except:
            print("Load failed. Start from Scratch.")
            return model, optimizer, scheduler, epoch
        print("Loaded self.model from epoch {}".format(self.epoch))
        return model, optimizer, scheduler, epoch
