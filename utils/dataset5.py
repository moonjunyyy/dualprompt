import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN, ImageFolder, mnist
import torchvision.transforms as transforms
from typing import Callable, Optional
from PIL import Image
import numpy as np

import torchvision.datasets as dd

class Dataset5(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        tf = transforms.Compose([transforms.Resize((224, 224))])
        super().__init__()
        self.cifar  = CIFAR10     (root, train, transform, target_transform, download)
        self.mnist  = MNIST       (root, train, transform, target_transform, download)
        self.fmnist = FashionMNIST(root, train, transform, target_transform, download)
        self.nmnist = ImageFolder (root + 'notMNIST1/notMNIST/notMNIST_large/Train/' if train else root + 'notMNIST1/notMNIST/notMNIST_large/Test/', transform, target_transform)
        self.svhn   = SVHN        (root, "train" if train else "test", transform, target_transform, download)

        self.classes = [str(i) for i in range(50)]
        self.targets = []
        for i, (img, cls) in enumerate(self.cifar):
            self.targets.append(cls)
        for i, (img, cls) in enumerate(self.mnist):
            self.targets.append(cls + 10)
        for i, (img, cls) in enumerate(self.fmnist):
            self.targets.append(cls + 20)
        for i, (img, cls) in enumerate(self.nmnist):
            self.targets.append(cls + 30)
        for i, (img, cls) in enumerate(self.svhn):
            self.targets.append(cls + 40)

    def __getitem__(self, index):
        target = self.targets[index]
        if target in [i for i in range(0,10)]:
            return self.cifar.__getitem__(index)[0], self.targets[index]
        elif target in [i for i in range(10,20)]:
            return self.mnist.__getitem__(index - len(self.cifar))[0].expand(3,-1,-1), self.targets[index]
        elif target in [i for i in range(20,30)]:
            return self.fmnist.__getitem__(index - len(self.cifar) - len(self.mnist))[0].expand(3,-1,-1), self.targets[index]
        elif target in [i for i in range(30,40)]:
            return self.nmnist.__getitem__(index - len(self.cifar) - len(self.mnist) - len(self.fmnist))[0].expand(3,-1,-1), self.targets[index]
        else:
            return self.svhn.__getitem__(index - len(self.cifar) - len(self.mnist) - len(self.fmnist) - len(self.nmnist))[0], self.targets[index]

    def __len__(self):
        return len(self.targets)