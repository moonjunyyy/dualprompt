from typing import Callable, Optional

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

class TinyImageNet(ImageFolder):
    def __init__(self, 
                 root             : str, 
                 train            : bool, 
                 transform        : Optional[Callable] = None, 
                 target_transform : Optional[Callable] = None, 
                 download         : bool = False
                 ) -> None:
        self.path = root + '/tiny-imagenet-200/'

        if train:
            super().__init__(self.path + "train", transform=transforms.ToTensor() if transform is None else transform, target_transform=target_transform)
            self.classes = []
            with open(self.path + "wnids.txt", 'r') as f:
                for id in f.readlines():
                    self.classes.append(id.split("\n")[0])
            self.class_to_idx = {clss: idx for idx, clss in enumerate(self.classes)}
            self.targets = []
            for idx, (path, _) in enumerate(self.samples):
                self.samples[idx] = (path, self.class_to_idx[path.split("/")[-3]])
                self.targets.append(self.class_to_idx[path.split("/")[-3]])

        else:
            super().__init__(self.path + "val", transform=transforms.ToTensor() if transform is None else transform, target_transform=target_transform)
            self.classes = []
            with open(self.path + "wnids.txt", 'r') as f:
                for id in f.readlines():
                    self.classes.append(id.split("\n")[0])
            self.class_to_idx = {clss: idx for idx, clss in enumerate(self.classes)}
            self.targets = []
            with open(self.path + "val/val_annotations.txt", 'r') as f:
                file_to_idx = {line.split('\t')[0] : self.class_to_idx[line.split('\t')[1]] for line in f.readlines()}
                for idx, (path, _) in enumerate(self.samples):
                    self.samples[idx] = (path, file_to_idx[path.split("/")[-1]])
                    self.targets.append(file_to_idx[path.split("/")[-1]])
