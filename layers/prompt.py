import torch
import torch.nn as nn
import torch.nn.functional as F


class Prompt(nn.Module):
    def __init__(self,
                 pool_size            : int,
                 selection_size       : int,
                 prompt_len           : int,
                 dimention            : int,
                 _batchwise_selection : bool = True,
                 _mixed_prompt_order  : bool = False,
                 _mixed_prompt_token  : bool = False,
                 **kwargs) -> None:
        super().__init__()

        self.pool_size      = pool_size
        self.selection_size = selection_size
        self.prompt_len     = prompt_len
        self.dimention      = dimention
        self._batchwise_selection = _batchwise_selection
        self._mixed_prompt_order  = _mixed_prompt_order
        self._mixed_prompt_token  = _mixed_prompt_token

        self.key     = nn.Parameter(torch.randn(pool_size, dimention, requires_grad= True))
        self.prompts = nn.Parameter(torch.randn(pool_size, prompt_len, dimention, requires_grad= True))
        
        torch.nn.init.uniform_(self.key,     -1, 1)
        torch.nn.init.uniform_(self.prompts, -1, 1)

        self.register_buffer('frequency', torch.ones (pool_size))
        self.register_buffer('counter',   torch.zeros(pool_size))
    
    def forward(self, query : torch.Tensor, **kwargs) -> torch.Tensor:

        B, D = query.shape
        assert D == self.dimention, f'Query dimention {D} does not match prompt dimention {self.dimention}'

        # Select prompts
        match = F.cosine_similarity(query.unsqueeze(1), self.key, dim=-1)
        if self.training:
            topk = match * F.normalize(self.frequency.reciprocal(), p=1, dim=-1)
        else:
            topk = match
        _ ,topk = topk.topk(self.selection_size, dim=-1, largest=True, sorted=True)

        # Batch-wise prompt selection
        if self._batchwise_selection:
            idx, counts = topk.unique(sorted=True, return_counts=True)
            _,   mosts  = counts.topk(self.selection_size, largest=True, sorted=True)
            topk = idx[mosts].clone().expand(B, -1)

        selection   = self.prompts[topk].clone()
        simmilarity = match.gather(1, topk).clone()

        # Mixed order prompt selection
        if self._mixed_prompt_order and self._mixed_prompt_token:
            selection = selection.reshape(B, -1, D)
            selection = selection[:, torch.randperm(self.selection_size * self.prompt_len)].clone()
            selection = selection.reshape(B, self.selection_size, self.prompt_len, D)
        elif self._mixed_prompt_order:
            selection = selection[:, torch.randperm(self.selection_size)].clone()
        elif self._mixed_prompt_order:
            selection = selection[:, :, torch.randperm(self.prompt_len)].clone()

        if self.training:
            self.counter += torch.bincount(topk.contiguous().view(-1), minlength = self.pool_size)
        return simmilarity, selection
    
    def update(self):
        self.frequency += self.counter
        self.counter   *= 0
        return self.frequency
