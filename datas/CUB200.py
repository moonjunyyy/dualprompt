import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder, ImageNet
import torchvision.transforms as transforms
from torch.utils.data import random_split
from typing import Callable, Optional

class CUB200(Dataset):
    def __init__(self, 
                 root             : str, 
                 train            : bool, 
                 transform        : Optional[Callable] = None, 
                 target_transform : Optional[Callable] = None, 
                 download         : bool = False
                 ) -> None:
        super().__init__()
        self.dataset = ImageFolder(root + '/CUB200-2011/images', transform, target_transform)
        train, test = random_split(self.dataset, [80, 20], generator=torch.Generator().manual_seed(42))
        self.dataset = train if train else test
        self.classes = self.dataset.classes
        self.targets = self.dataset.targets
        pass
    
    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)