from typing import Callable, Optional

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import (CIFAR10, MNIST, SVHN, FashionMNIST,
                                  ImageFolder)

from datas.NotMNIST import NotMNIST


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
        self.nmnist = NotMNIST    (root, train, transform, target_transform, download)
        self.svhn   = SVHN        (root, "train" if train else "test", transform, target_transform, download)

        self.classes = [str(i) for i in range(50)]
        self.targets = []

        self.first  = 0
        self.second = 0
        self.third  = 0
        self.fourth = 0
        self.fifth  = 0

        for cls in self.cifar.targets:
            self.targets.append(cls)
        self.first = len(self.targets)
        for cls in self.mnist.targets:
            self.targets.append(cls + 10)
        self.second = len(self.targets)
        for cls in self.fmnist.targets:
            self.targets.append(cls + 20)
        self.third = len(self.targets)
        for cls in self.nmnist.targets:
            self.targets.append(cls + 30)
        self.fourth = len(self.targets)
        for img, cls in self.svhn:
            self.targets.append(cls + 40)
        self.fifth = len(self.targets)

    def __getitem__(self, index):
        target = self.targets[index]
        if target in [i for i in range(0,10)]:
            return self.cifar .__getitem__(index)[0],                               self.targets[index]
        elif target in [i for i in range(10,20)]:
            return self.mnist .__getitem__(index - self.first )[0].expand(3,-1,-1), self.targets[index]
        elif target in [i for i in range(20,30)]:
            return self.fmnist.__getitem__(index - self.second)[0].expand(3,-1,-1), self.targets[index]
        elif target in [i for i in range(30,40)]:
            return self.nmnist.__getitem__(index - self.third )[0].expand(3,-1,-1), self.targets[index]
        else:
            return self.svhn  .__getitem__(index - self.fourth)[0],                 self.targets[index]

    def __len__(self):
        return len(self.targets)
