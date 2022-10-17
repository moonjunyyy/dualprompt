from typing import TypeVar

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Prompt import Prompt

T = TypeVar('T', bound = 'nn.Module')

class ContrastiveL2P(nn.Module):
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
                 _batchwise_selection  : bool = True,
                 _diversed_selection   : bool = True,
                 _update_per_iter      : bool = False,
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
        self.zetta = zetta
        self._batchwise_selection = _batchwise_selection
        self._update_per_iter = _update_per_iter
        self.class_num = class_num

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

        self.register_buffer('simmilarity',   torch.zeros(1))
        self.register_buffer('unsimmilarity', torch.zeros(1))
        self.register_buffer('mask',  torch.zeros(class_num))
        self.register_buffer('const',         torch.zeros(1))

        self.prompt.key.requires_grad = False
        
    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.backbone.patch_embed(inputs)
        
        B, N, D = x.size()
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        t = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(t + self.backbone.pos_embed)

        q = self.backbone.blocks(x)
        q = self.backbone.norm(q)[:, 0].clone()

        s, us, p, up = self.prompt(q)

        p  =  p.contiguous().view(B, self.selection_size * self.prompt_len, D)
        p  =  p + self.backbone.pos_embed[:,0].expand(self.selection_size * self.prompt_len, -1)
        up = up.contiguous().view(B, self.selection_size * self.prompt_len, D)
        up = up + self.backbone.pos_embed[:,0].expand(self.selection_size * self.prompt_len, -1)

        x = self.backbone.pos_drop(t + self.backbone.pos_embed)
        sx = torch.cat((x[:,0].unsqueeze(1),  p, x[:,1:]), dim=1)
        # ux = torch.cat((x[:,0].unsqueeze(1), up, x[:,1:]), dim=1)

        sx = self.backbone.blocks(sx)
        sx = self.backbone.norm(sx)
        sx = sx[:, 1:self.selection_size * self.prompt_len + 1].clone()

        # ux = self.backbone.blocks(ux)
        # ux = self.backbone.norm(ux)
        # ux = ux[:, 1:self.selection_size * self.prompt_len + 1].clone()

        # const = torch.cat((sx, ux), dim=1)
        # const = const @ const.transpose(-1,-2)
        # const = ((const - const.max())/ self.tau).exp() 
        # const = const[:, :self.selection_size, :self.selection_size].sum(-1).sum(-1) / const.sum(-1).sum(-1)
        # const = -(const + 1e-6).log().sum()
        # self.const = const

        # sim = torch.cat((1-s, 1-us), dim=1)

        cossim      = (1 - s).unsqueeze(-1)
        sinsim      = (1 - cossim * cossim).sqrt()
        self.const  = (cossim @ cossim.transpose(-1,-2) - sinsim @ sinsim.transpose(-1,-2)).sum()
        self.simmilarity   = s.sum() #-( ((1-s)/self.tau).exp().sum(-1) / (((1-s)/self.tau).exp().sum(-1) + ((1-us)/self.tau).exp().sum(-1)) + 1e-6).log().sum() #s.sum() + 
        self.unsimmilarity = (1-us).sum()

        sx = sx.mean(dim=1)
        x = self.backbone.head(sx)

        if self._update_per_iter:
            self.prompt.update()

        if self.training:
            x = x + self.mask
        return x
    
    def loss_fn(self, output, target):
        B, C = output.size()
        return F.cross_entropy(output, target) + self.lambd * self.simmilarity + self.xi * self.unsimmilarity + self.zetta * self.const 

    def _convert_train_task(self, task : torch.Tensor, **kwargs):
        self.mask += -torch.inf
        self.mask[task.to(self.mask.device)] = 0
        return self.prompt.update()

    def train(self: T, mode: bool = True, **kwargs) -> T:
        ten = super().train(mode)
        self.backbone.eval()
        return ten
