import torch
import torch.nn as nn
import torch.nn.functional as F


class Prompt(nn.Module):
    def __init__(self, pool_size : int, selection_size : int, prompt_len : int, dimention : int, batchwise_selection : bool = True, **kwargs) -> None:
        super().__init__()

        self.pool_size      = pool_size
        self.selection_size = selection_size
        self.prompt_len     = prompt_len
        self.dimention      = dimention
        self.batchwise_selection = batchwise_selection

        self.key     = nn.Parameter(torch.randn(pool_size, dimention, requires_grad= True))
        self.prompts = nn.Parameter(torch.randn(pool_size, prompt_len, dimention, requires_grad= True))

        torch.nn.init.uniform_(self.key,     -1, 1)
        torch.nn.init.uniform_(self.prompts, -1, 1)

        self.register_buffer('frequency', torch.ones (pool_size))
        self.register_buffer('counter',   torch.zeros(pool_size))
    
    def forward(self, query : torch.Tensor, **kwargs) -> torch.Tensor:

        B, D = query.shape
        assert D == self.dimention, f'Query dimention {D} does not match prompt dimention {self.dimention}'

        match = F.cosine_similarity(query.unsqueeze(1), self.key, dim=-1)
        if self.training:
            topk = match * F.normalize(1 / self.frequency, p=1, dim=-1)
        else:
            topk = match
        _ ,topk = topk.topk(self.selection_size, dim=-1, largest=True, sorted=True)
        if self.batchwise_selection:
            idx, counts = topk.unique(sorted=True, return_counts=True)
            _, mosts = counts.topk(self.selection_size, largest=True, sorted=True)
            topk = idx[mosts].clone().unsqueeze(0).expand(B, -1)
        if self.training:
            self.counter += torch.bincount(topk.contiguous().view(-1), minlength = self.pool_size)
        return match.gather(1, topk).clone(), self.prompts[topk].clone()
    
    def update(self):
        self.frequency += self.counter
        self.counter   *= 0
        return self.frequency