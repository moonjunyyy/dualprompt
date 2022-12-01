# Wraping for the SHVN dataset

from typing import Callable, Optional

from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST

class FashionMNIST(Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__()
        self.dataset = FashionMNIST(root, train, transform, target_transform, download)
        
        self.classes = [str(i) for i in range(10)]
        self.targets = []
        for cls in self.dataset.labels:
            self.targets.append(int(cls))

    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(index)
        return image.expand(3,-1,-1), label
        
    def __len__(self):
        return len(self.dataset)