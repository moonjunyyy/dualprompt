from itertools import count
import torch
import torch.nn as nn
import torch.nn.functional as F

class Prompt(nn.Module):
    def __init__(self, pool_size : int, selection_size : int, prompt_len : int, dimention : int, *args, **kwargs):
        super(Prompt, self).__init__()

        self.pool_size      = pool_size
        self.selection_size = selection_size
        self.prompt_len     = prompt_len
        self.dimention      = dimention

        self.key            = nn.Parameter(torch.randn(pool_size,             dimention, requires_grad=True))
        self.prompt         = nn.Parameter(torch.randn(pool_size, prompt_len, dimention, requires_grad=True))

        self.key    = nn.init.uniform_(self.key,    -1, 1)
        self.prompt = nn.init.uniform_(self.prompt, -1, 1)

        self.frequency      = torch.ones (pool_size)
        self.counter        = torch.zeros(pool_size)

    def forward(self, query : torch.Tensor, *args, **kwargs):
        
        self.frequency = self.frequency.to(query.device)
        self.counter   = self.counter.to  (query.device)
        self.key       = self.key.to      (query.device)
        self.prompt    = self.prompt.to   (query.device)

        #self.key    = nn.Parameter(F.normalize(self.key,    p=2, dim=-1))
        #self.prompt = nn.Parameter(F.normalize(self.prompt, p=2, dim=-1))

        match = (1 - F.cosine_similarity(query.unsqueeze(1), self.key, dim = -1))

        if self.training:
            topk    = match
            #topk    = (match * F.normalize(self.frequency, p=1, dim=-1))
        else :
            topk    = match

        _, topk     = topk.topk(self.selection_size, dim = -1, largest = False, sorted = True)
        idx, counts = topk.unique(sorted=True, return_counts=True)
        _, mosts    = counts.topk(self.selection_size, largest = True, sorted = True)
        topk        = idx[mosts]
        if self.training:
            self.counter += topk.bincount(minlength = self.pool_size)

        return match.gather(-1, topk.unsqueeze(0)), self.prompt[topk].unsqueeze(0).expand(query.size(0),-1,-1,-1)

    def update(self):
        self.frequency += self.counter
        self.counter   *= 0
        return self.frequency