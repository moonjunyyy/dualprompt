from asyncio import tasks
from distutils.command.build import build
import torch
import torch.distributed as dist
import math

class RASampler(torch.utils.data.Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, num_repeats: int = 3):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if num_repeats < 1:
            raise ValueError("num_repeats should be greater than 0")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * self.num_repeats / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(start=0, end=len(self.dataset))

        # add extra samples to make it evenly divisible
        indices = torch.repeat_interleave(indices, repeats=self.num_repeats, dim=0).tolist()
        padding_size: int = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
    
class CILSampler(torch.utils.data.Sampler):
    r"""Sampler that Samples a subset of the dataset for distributed CIL training.
    Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    or works like as common sampler in a non-distributed setting.
    Heavily based on RASampler by Facebook.
    """
    def __init__(self, dataset, num_tasks = 1, num_replicas=None, rank=None, shuffle=True, num_repeats: int = 3, seed: int = 0):

        if num_replicas is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            num_replicas = dist.get_world_size()
        if rank is not None:
            if not dist.is_available():
                raise RuntimeError("Distibuted package is not available, but you are trying to use it.")
            rank = dist.get_rank()
        if num_repeats < 1:
            num_repeats = 1

        self.distributed = num_replicas is not None and rank is not None
        self.dataset = dataset
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0
        self.num_repeats = num_repeats
        self.epoch = 0
        self.shuffle = shuffle
        self.num_tasks = num_tasks
        self.task = 0
        self.seed = seed

        self.g = torch.Generator()
        self.g.manual_seed(self.seed)

        stale = len(self.dataset.classes) - len(self.dataset.classes) % self.num_tasks
        self.taskids = torch.randperm(len(self.dataset.classes), generator = self.g)
        self.taskids = self.taskids[:stale].reshape(self.num_tasks, -1)
        self.build()
        
    def build(self):
        self.g.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            idx = torch.randperm(len(self.dataset) * self.num_replicas, generator=self.g) % len(self.dataset)
        else:
            idx = torch.arange(len(self.dataset) * self.num_replicas) % len(self.dataset)
        sel = (torch.tensor(self.dataset.targets)[idx] == self.taskids[self.task].unsqueeze(-1)).sum(0).nonzero()
        self.indices = idx[sel].squeeze()
        if self.distributed:
            self.num_samples = int(len(self.indices.tolist()) // self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas  
            self.num_selected_samples = int(len(self.indices.tolist()) // self.num_replicas)
        else:
            self.num_samples = int(len(self.indices.tolist()))
            self.total_size = self.num_samples
            self.num_selected_samples = int(len(self.indices.tolist()))

    def __iter__(self):
        self.build()
        if self.distributed:
            # subsample
            self.indices = self.indices[self.rank:self.total_size:self.num_replicas]
            assert len(self.indices) == self.num_samples
            return iter(self.indices[:self.num_selected_samples])
        else:
            return iter(self.indices)

    def __len__(self):
        return self.num_selected_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_task(self, task):
        if task >= self.num_tasks or task < 0:
            raise ValueError("Task index out of range")
        self.task = task
    
    def get_task(self):
        return self.taskids[self.task]
        