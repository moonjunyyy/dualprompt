from statistics import mode
from typing import TypeVar

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

T = TypeVar('T', bound = 'nn.Module')

class CertViT(nn.Module):
    def __init__(self,
                 backbone_name   : str   = None,
                 class_num       : int   = 100,
                 reserve_rate    : float = 0.7,
                 selection_layer : tuple = (3,),
                 _mode           : bool  = False,  
                 **kwargs):

        super().__init__()
        
        self._mode = _mode
        if backbone_name is None:
            raise ValueError('backbone_name must be specified')
        self.reserve_rate = reserve_rate
        self.register_buffer('selection_layer', torch.tensor(selection_layer))

        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=class_num))
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.head.weight.requires_grad = True
        self.backbone.head.bias.requires_grad = True
        
    def feature_forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.backbone.patch_embed(inputs)
        B, N, C = x.size()
        for n, block in enumerate(self.backbone.blocks):
            r  = x
            x = block.norm1(x)
          
            msa   = block.attn
            qkv = msa.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            
            layer = ((self.selection_layer == n).nonzero()).squeeze()
            if layer.numel() != 0:
                if self._mode:
                    # Who Cares Me (Key) Filter
                    uncertainty = N / (attn + 1).sum(dim = -2)
                else:
                    # Where to See (Queue) Filter
                    uncertainty = N / (attn + 1).sum(dim = -1)
                uncertainty, idx = uncertainty.topk(int(N * self.reserve_rate), dim = -1, sorted=False)
                attn = attn[idx[0],idx[1],idx[1]]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            x = r + block.drop_path(x)
            x = x + block.drop_path(block.mlp(block.norm2(x)))
        x = self.backbone.norm(x)
        return x

    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.feature_forward(inputs)
        x = self.backbond.head(x)
        return x