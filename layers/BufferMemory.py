import torch
import torch.nn as nn
import torch.nn.functional as F

class BufferMemory(nn.Module):
    def __init__(self,
                 memory_size   : int,
                 *tensor_size  : int,
                 reduce_method : str = 'mean',
                 **kwargs) -> None:
        super().__init__()
        
        self.memory_size = memory_size
        self.memory = torch.zeros(memory_size, *tensor_size)
        self.register_buffer('index', torch.zeros(1, dtype=torch.long))
        self.reduce_item = self.reduce_distance() if reduce_method == 'mean' else self.reduce_unlike() if reduce_method == 'unlike' else self.reduce_cos_sim()
        self.reduce = reduce_method

    def update(self, tensor : torch.Tensor) -> None:
        if self.index >= self.memory_size:
            self.reduce_item()
        else:
            self.memory[self.index] = tensor
            self.index = (self.index + 1) % self.memory_size

    def get_item(self, index : int) -> torch.Tensor:
        return self.memory[index]

    def set_item(self, index : int, tensor : torch.Tensor) -> None:
        self.memory[index] = tensor

    def get_memory(self) -> torch.Tensor:
        return self.memory
    
    def get_index(self) -> int:
        return self.index.item()

    def reduce_cos_sim(self, tensor : torch.Tensor) -> torch.Tensor:
        return torch.cosine_similarity(tensor, self.memory, dim=1)

    def reduce_distance(self) -> None:
        pass

    def reduce_unlike(self) -> None:
        pass