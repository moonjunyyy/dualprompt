from typing import TypeVar

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Prompt import Prompt

T = TypeVar('T', bound = 'nn.Module')

class GausskeyL2P(nn.Module):
    def __init__(self,
                 pool_size      : int   = 10,
                 selection_size : int   = 5,
                 prompt_len     : int   = 5,
                 class_num      : int   = 100,
                 backbone_name  : str   = None,
                 lambd          : float = 0.5,
                 tau            : float = 0.5,
                 xi             : float = 0.1,
                 zetta          : float = 0.1,
                 query_mem_size : int   = 10,
                 _batchwise_selection  : bool = True,
                 _diversed_selection   : bool = True,
                 _update_per_iter      : bool = False,
                 **kwargs):

        super().__init__()
        
        if backbone_name is None:
            raise ValueError('backbone_name must be specified')
        if pool_size < selection_size:
            raise ValueError('pool_size must be larger than selection_size')

        self.pool_size      = pool_size
        self.prompt_len     = prompt_len
        self.selection_size = selection_size
        self.query_mem_size = query_mem_size
        self.lambd = lambd
        self.tau   = tau
        self.xi    = xi
        self.zetta = zetta
        self._batchwise_selection = _batchwise_selection
        self._update_per_iter     = _update_per_iter
        self.class_num      = class_num

        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=class_num))
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.head.weight.requires_grad = True
        self.backbone.head.bias.requires_grad = True

        self.prompt = Prompt(
            pool_size,
            selection_size,
            prompt_len,
            self.backbone.num_features,
            _diversed_selection  = _diversed_selection,
            _batchwise_selection = _batchwise_selection,
            _get_unsimmilarity   = True)

        _tmp = torch.rand((pool_size, self.backbone.num_features, self.backbone.num_features)) * 2
        self.register_buffer('variance',      (_tmp + _tmp.transpose(-1,-2)))
        self.register_buffer('inv_variance',  torch.linalg.inv(self.variance))
        self.register_buffer('mask',          torch.zeros(class_num))
        
        for num in range(pool_size):
            self.register_buffer(f'query_memory_{num}',  torch.ones((query_mem_size, self.backbone.num_features)) * torch.inf)
        self.prompt.key.requires_grad = False
    
    def update_query_memory(self, query : torch.Tensor, idx : torch.Tensor):
        B, C = query.shape
        
        for batch in range(idx.size(0)):
            query_memory = getattr(self, f'query_memory_{idx[batch]}')
            query_memory = torch.cat((query_memory, query[batch].unsqueeze(0)), dim = 0)

        for num in range(self.pool_size):
            query_memory = getattr(self, f'query_memory_{num}')
            query_memory = query_memory[-self.query_mem_size:]
            setattr(self, f'query_memory_{num}', query_memory)

    def gaussian(self, input : torch.Tensor):
        B, D = input.shape
        mean = self.prompt.key.detach() # P,D
        coff = (torch.tensor([2 * torch.pi], device=mean.device).pow(self.backbone.num_features / 2) * self.variance.norm(dim = (1,2)).sqrt()).reciprocal()
        return ((input.unsqueeze(1) - mean).unsqueeze(-2) @ self.inv_variance @ (input.unsqueeze(1) - mean).unsqueeze(-1) / -2).squeeze().exp() / coff
        # B, P, D, 1 @ P, D, D @ B, P, D, 1= B, P, 1, 1
    
    def update_keys(self, input : torch.Tensor, idx : torch.Tensor):

        B, D = input.shape
        mean = self.prompt.key.detach() # P, D
        P, D = mean.shape
        mask = torch.scatter(torch.zeros(B, P, device=mean.device), -1, idx[:,:3], torch.ones(B, P, device=mean.device)) # top3 selection
        # B, P
        counts = torch.bincount(idx[:,:3].contiguous().view(-1), minlength=P)
        meanL1 = (mask.unsqueeze(-1) * input.unsqueeze(1)).transpose(0,1).sum(1)
        # B, D
        meanL2 = (mask.unsqueeze(-1).unsqueeze(-1) * (input.unsqueeze(-1) @ input.unsqueeze(1)).unsqueeze(1)).transpose(0,1).sum(1)

        # var  = (self.variance + (mean.unsqueeze(-1) @ mean.unsqueeze(1))) * (self.prompt.frequency.log() + 1).unsqueeze(-1).unsqueeze(-1) + meanL2
        # mean = mean * (self.prompt.frequency.log() + 1).unsqueeze(-1) + meanL1
        # self.prompt.key = nn.Parameter(mean / (self.prompt.frequency.log() + 2).unsqueeze(-1), requires_grad=False)
        # self.variance = (var / (self.prompt.frequency.log() + 2).unsqueeze(-1).unsqueeze(-1)) - self.prompt.key.unsqueeze(-1) @ self.prompt.key.unsqueeze(-2)
        # self.inv_variance = torch.linalg.inv(self.variance)
        # self.prompt.frequency += counts

        var  = (self.variance + (mean.unsqueeze(-1) @ mean.unsqueeze(1))) * self.prompt.frequency.unsqueeze(-1).unsqueeze(-1) + meanL2
        mean = mean * self.prompt.frequency.unsqueeze(-1) + meanL1
        self.prompt.frequency += counts
        self.prompt.key = nn.Parameter(mean / self.prompt.frequency.unsqueeze(-1), requires_grad=False)
        self.variance = (var / self.prompt.frequency.unsqueeze(-1).unsqueeze(-1)) - self.prompt.key.unsqueeze(-1) @ self.prompt.key.unsqueeze(-2)
        self.inv_variance = torch.linalg.inv(self.variance)
        

    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.backbone.patch_embed(inputs)
        
        B, N, D = x.size()
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        t = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(t + self.backbone.pos_embed)

        q = self.backbone.blocks(x)
        q = self.backbone.norm(q)[:, 0].clone()

        prob = self.gaussian(q)
        prob = prob * F.normalize(self.prompt.frequency.max() - self.prompt.frequency, dim=-1) #Diversed Selection
        _ ,topk = prob.topk(self.selection_size, dim=-1, largest=False, sorted=True)
        p = self.prompt.prompts.repeat(B, 1, 1, 1).gather(1, topk.unsqueeze(-1).unsqueeze(-1).expand(B, -1, self.prompt_len, self.backbone.num_features).clone())
        # if self.training : 
        #     self.update_keys(q, topk)

        p  =  p.contiguous().view(B, self.selection_size * self.prompt_len, D)
        p  =  p + self.backbone.pos_embed[:,0].expand(self.selection_size * self.prompt_len, -1)

        x = self.backbone.pos_drop(t + self.backbone.pos_embed)
        sx = torch.cat((x[:,0].unsqueeze(1),  p, x[:,1:]), dim=1)

        sx = self.backbone.blocks(sx)
        sx = self.backbone.norm(sx)
        sx = sx[:, 1:self.selection_size * self.prompt_len + 1].clone()

        sx = sx.mean(dim=1)
        x = self.backbone.head(sx)

        if self._update_per_iter:
            self.prompt.update()

        if self.training:
            x = x + self.mask
        return x
    
    def loss_fn(self, output, target):
        B, C = output.size()
        return F.cross_entropy(output, target)

    def _convert_train_task(self, task : torch.Tensor, **kwargs):
        self.mask += -torch.inf
        self.mask[task.to(self.mask.device)] = 0
        return self.prompt.update()

    def train(self: T, mode: bool = True, **kwargs) -> T:
        ten = super().train(mode)
        self.backbone.eval()
        return ten
