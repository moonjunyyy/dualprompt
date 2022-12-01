from typing import Callable, Optional

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import SVHN


class _DatasetCopy(Dataset):
    def __init__(self, dataset : Dataset|Subset, transform: Optional[Callable] = None) -> None:
        super().__init__()
        dl = DataLoader(dataset, 256, shuffle=False, num_workers=4)
        for n, (data, target) in enumerate(dl):
            if n == 0:
                self.data    = data
                self.targets = target
                continue
            self.data = torch.concat((self.data, data), dim = 0)
            self.targets = torch.concat((self.targets, target), dim = 0)
        if isinstance(dataset, Subset) : self.classes = dataset.dataset.classes
        elif isinstance(dataset, SVHN) : self.classes = [i for i in range(10)]
        else:                            self.classes = dataset.classes
        self.transform = transform

    def __getitem__(self, index):
        t = transforms.ToPILImage()
        return self.transform(t(self.data[index])), self.targets[index]

    def __len__(self):
        return len(self.data)