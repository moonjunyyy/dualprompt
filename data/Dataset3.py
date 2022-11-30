from typing import Callable, Optional

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import StanfordCars, Flowers102
from data.CUB200 import CUB200

class Dataset3(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__()
        self.cub    = CUB200(root, train, transform, target_transform, download)
        self.car    = StanfordCars(root, "train" if train else "test", transform, target_transform, download)
        self.flower = Flowers102(root, "train" if train else "test", transform, target_transform, download)

        self.classes = [str(i) for i in range(498)]
        self.targets = []

        self.first  = 0
        self.second = 0
        self.third  = 0

        for cls in self.cub.targets:
            self.targets.append(int(cls))
        self.first = len(self.targets)
        for _, cls in self.car._samples:
            self.targets.append(int(cls + 200))
        self.second = len(self.targets)
        for cls in self.flower._labels:
            self.targets.append(int(cls + 396))
        self.third = len(self.targets)

    def __getitem__(self, index):
        target = self.targets[index]
        if target >= 0 and target < 200:
            item = self.cub.__getitem__(index)[0]
        elif target >= 10 and target < 396:
            item = self.car.__getitem__(index - self.first)[0]
        else:
            item = self.flower.__getitem__(index - self.second)[0]
        return item, target

    def __len__(self):
        return len(self.targets)
