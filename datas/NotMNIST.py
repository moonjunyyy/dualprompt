from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from typing import Callable, Optional

class NotMNIST(Dataset):
    def __init__(self, 
                 root             : str, 
                 train            : bool, 
                 transform        : Optional[Callable] = None, 
                 target_transform : Optional[Callable] = None, 
                 download         : bool = False
                 ) -> None:
        super().__init__()
        self.dataset = ImageFolder(root + '/notMNIST_large/Train/' if train 
                              else root + '/notMNIST_large/Test/',
                              transform, target_transform)
        self.classes = self.dataset.classes
        self.targets = self.dataset.targets
        pass
    
    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)