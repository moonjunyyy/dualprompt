from torch.utils.data import Dataset, Subset, DataLoader
import torch

class _Subset(Dataset):
    def __init__(self, subset : Subset) -> None:
        super().__init__()
        dl = DataLoader(subset, 1024, shuffle=False, num_workers=4)
        for n, (data, target) in enumerate(dl):
            if n == 0:
                self.data    = data
                self.targets = target
                continue
            self.data = torch.concat((self.data, data), dim = 0)
            self.targets = torch.concat((self.targets, target), dim = 0)
        self.classes = subset.dataset.classes

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)