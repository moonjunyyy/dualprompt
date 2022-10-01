import torch
import torch.nn as nn
import torch.nn.functional as F


class SimUnsimPrompt(nn.Module):
    def __init__(self,
                 pool_size            : int,
                 selection_size       : int,
                 prompt_len           : int,
                 dimention            : int,
                 _diversed_selection  : bool = True,
                 _batchwise_selection : bool = True,
                 **kwargs) -> None:
        super().__init__()

        self.pool_size      = pool_size
        self.selection_size = selection_size
        self.prompt_len     = prompt_len
        self.dimention      = dimention
        self._diversed_selection  = _diversed_selection
        self._batchwise_selection = _batchwise_selection

        self.key     = nn.Parameter(torch.randn(pool_size, dimention, requires_grad= True))
        self.prompts = nn.Parameter(torch.randn(pool_size, prompt_len, dimention, requires_grad= True))
        
        torch.nn.init.uniform_(self.key,     -1, 1)
        torch.nn.init.uniform_(self.prompts, -1, 1)

        self.register_buffer('frequency', torch.ones (pool_size))
        self.register_buffer('counter',   torch.zeros(pool_size))
        self.register_buffer('topk', torch.zeros(1))
        self.register_buffer('nonk', torch.zeros(1))
    
    def forward(self, query : torch.Tensor, **kwargs) -> torch.Tensor:

        B, D = query.shape
        assert D == self.dimention, f'Query dimention {D} does not match prompt dimention {self.dimention}'

        # Select prompts
        match = F.cosine_similarity(query.unsqueeze(1), self.key, dim=-1)
        if self.training and self._diversed_selection:
            topk = match * F.normalize(self.frequency.reciprocal(), p=1, dim=-1)
        else:
            topk = match
        _ ,topk = topk.topk(self.selection_size, dim=-1, largest=True, sorted=True)

        # Batch-wise prompt selection
        if self._batchwise_selection:
            idx, counts = topk.unique(sorted=True, return_counts=True)
            _,  mosts  = counts.topk(self.selection_size, largest=True, sorted=True)
            topk = idx[mosts].clone().expand(B, -1)

        if self.training:
            self.counter += torch.bincount(topk.contiguous().view(-1), minlength = self.pool_size)

        nonk = torch.ones((B, self.pool_size), device=self.prompts.device, dtype=torch.long)
        nonk = nonk.scatter(1, topk, torch.zeros_like(nonk))
        nonk = nonk.nonzero(as_tuple=True)[1].reshape(topk.size())

        self.topk = topk
        self.nonk = nonk

        selection   = self.prompts.repeat(B, 1, 1, 1).gather(1, topk.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.prompt_len, self.dimention))
        simmilarity   = match.gather(1, topk)
        unsimmilarity = match.gather(1, nonk)

        return simmilarity, unsimmilarity, selection
    
    def update(self):
        self.frequency += self.counter
        self.counter   *= 0
        return self.frequency
