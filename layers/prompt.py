import torch
import torch.nn as nn
import torch.nn.functional as F

class Prompt(nn.Module):
    def __init__(self, pool_size : int, selection_size : int, prompt_len : int, dimention : int, batchwise_selection : bool = True, **kwargs):
        super(Prompt, self).__init__()

        self.pool_size      = pool_size
        self.selection_size = selection_size
        self.prompt_len     = prompt_len
        self.dimention      = dimention
        self.batchwise_selection = batchwise_selection

        self.key            = nn.Parameter(torch.randn(pool_size,             dimention, requires_grad=True))
        self.prompt         = nn.Parameter(torch.randn(pool_size, prompt_len, dimention, requires_grad=True))

        self.key    = nn.init.uniform_(self.key,    -1, 1)
        self.prompt = nn.init.uniform_(self.prompt, -1, 1)

        self.frequency      = torch.ones (pool_size)
        self.counter        = torch.zeros(pool_size)

    def forward(self, query : torch.Tensor, *args, **kwargs):
        self.to(query.device)
        match = F.cosine_similarity(query.unsqueeze(1), self.key, dim = -1)
        if self.training:
            topk    = match * F.normalize(1 / self.frequency, p=1, dim=-1)
        else :
            topk    = match
        _, topk     = topk.topk(self.selection_size, dim = -1, largest = True, sorted = True)

        if self.batchwise_selection:
            idx, counts = topk.unique(sorted=True, return_counts=True)
            _, mosts    = counts.topk(self.selection_size, largest = True, sorted = True)
            topk        = idx[mosts].unsqueeze(0).repeat(query.size()[0], 1)
        if self.training:
            self.counter += topk.view(-1).bincount(minlength = self.pool_size)

        return match[topk], self.prompt[topk]

    def update(self):
        self.frequency += self.counter
        self.counter   *= 0
        return self.frequency

    def to(self, device, **kwargs):
        self.frequency = self.frequency.to(device)
        self.counter   = self.counter.to  (device)
        self.key.to      (device)
        self.prompt.to   (device)
        return self