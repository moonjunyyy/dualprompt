from dataclasses import astuple
from typing import TypeVar

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans, DBSCAN

T = TypeVar('T', bound = 'nn.Module')

class Prompt_Pool(nn.Module):
    def __init__(self,
                pool_size          : int   = 10,
                selection_size     : int   = 5,
                prompt_len         : int   = 5,
                prompt_dim         : int   = 768,
                generate_threshold : float = 0.5,
                merge_threshold    : float = 0.5,
                **kwarg) -> None:
        super().__init__()

        self.pool_size          = pool_size
        self.selection_size     = selection_size
        self.prompt_len         = prompt_len
        self.prompt_dim         = prompt_dim
        self.generate_threshold = generate_threshold
        self.merge_threshold    = merge_threshold

        self.register_buffer('key',            torch.randn(1, prompt_dim))
        self.register_buffer('prompts',        torch.randn(1, prompt_len, prompt_dim))
        self.register_buffer('num_selections', torch.zeros(1))
        pass
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        # calculate cosine similarity between x and query
        distance = 1 - F.cosine_similarity(x.unsqueeze(1), self.key, dim = -1) # B, N
        # select distance above threshold
        sim, idx = distance.min(-1)
        self.add_prompt(x.reshape(-1, D))
        self.merge_prompt()
        
        distance = 1 - F.cosine_similarity(x.unsqueeze(1), self.key, dim = -1) # B, N
        # select topk most similar prompts
        distance, topk = distance.topk(self.selection_size, dim = -1, largest = False, sorted = True)
        prompt = self.prompts[topk]
        self.num_selections += topk.reshape(-1).bincount(minlength = self.pool_size)
        return distance, prompt
    
    def add_prompt(self, x : torch.Tensor) -> None:
        # add prompt to pool
        N, D = x.shape
        self.key = torch.cat((self.key, x), dim = 0)
        self.prompts = torch.cat((self.prompts, torch.randn(N, self.prompt_len, self.prompt_dim, device=self.key.device)), dim = 0)
        self.num_selections = torch.cat((self.num_selections, torch.zeros(N, device=self.key.device)), dim = 0)
        return

    def merge_prompt(self) -> None:
        # merge similar prompts in pool
        N, D = self.key.shape
        _np_key = self.key.cpu().numpy()
        _kMeans = KMeans(n_clusters = self.pool_size, random_state = 0).fit_predict(_np_key)
        for i in range(self.pool_size):
            mask = _kMeans == i
            self.key[i] = self.key[mask].mean(0)
            self.prompts[i] = self.prompts[mask].mean(0)
            self.num_selections[i] = self.num_selections[mask].sum()
        self.key = self.key[:self.pool_size]
        self.prompts = self.prompts[:self.pool_size]
        self.num_selections = self.num_selections[:self.pool_size]
        return

class DyL2P(nn.Module):
    def __init__(self,
                 pool_size      : int   = 10,
                 selection_size : int   = 5,
                 prompt_len     : int   = 5,
                 class_num      : int   = 100,
                 backbone_name  : str   = None,
                 lambd          : float = 0.5,
                 tau            : float = 0.5,
                 xi             : float = 0.1,
                 **kwargs):
        super().__init__()
        
        if backbone_name is None:
            raise ValueError('backbone_name must be specified')
        if pool_size < selection_size:
            raise ValueError('pool_size must be larger than selection_size')

        self.prompt_len     = prompt_len
        self.selection_size = selection_size
        self.lambd = lambd
        self.tau   = tau
        self.xi    = xi
        self.class_num            = class_num

        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=class_num))
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.head.weight.requires_grad = True
        self.backbone.head.bias.requires_grad   = True

        self.prompt = Prompt_Pool(pool_size, selection_size, prompt_len, self.backbone.num_features)

        self.register_buffer('simmilarity', torch.zeros(1))
        self.register_buffer('mask', torch.zeros(class_num))
    
    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.backbone.patch_embed(inputs)

        B, N, D = x.size()
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        t = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(t + self.backbone.pos_embed)

        q = self.backbone.blocks(x)
        q = self.backbone.norm(q)[:, 0].clone()

        s, p = self.prompt(q)
        self.simmilarity = s.sum()
        p = p.contiguous().view(B, self.selection_size * self.prompt_len, D)
        p = p + self.backbone.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)

        x = self.backbone.pos_drop(t + self.backbone.pos_embed)
        x = torch.cat((x[:,0].unsqueeze(1), p, x[:,1:]), dim=1)

        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)

        x = x[:, 1:self.selection_size * self.prompt_len + 1].clone()
        x = x.mean(dim=1)
        x = self.backbone.head(x)

        if self.training:
            x = x + self.mask
        return x
    
    def loss_fn(self, output, target):
        B, C = output.size() 
        return F.cross_entropy(output, target) - self.lambd * self.simmilarity

    def _convert_train_task(self, task : torch.Tensor, **kwargs):
        self.mask += -torch.inf
        self.mask[task.to(self.mask.device)] = 0
        return self.prompt.num_selections

    def train(self: T, mode: bool = True, **kwargs) -> T:
        ten = super().train(mode)
        self.backbone.eval()
        return ten
