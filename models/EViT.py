from typing import TypeVar

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

T = TypeVar('T', bound = 'nn.Module')

class EViT(nn.Module):
    def __init__(self,
                 backbone_name   : str   = None,
                 class_num       : int   = 100,
                 reserve_rate    : float = 0.7,
                 selection_layer : tuple = (3,),
                 **kwargs):

        super().__init__()
        
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
        B, L, D = x.size()
        for n, block in enumerate(self.backbone.blocks):
            x = block(x)
            layer = ((self.selection_layer == n).nonzero()).squeeze()
            if layer.numel() != 0:
                cls_tkn = x[:,0].unsqueeze(1)
                sim = F.cosine_similarity(cls_tkn, x[:,1:], dim = -1)
                sim, idx = sim.topk(int(L * self.reserve_rate), dim = -1, sorted=False)
                x = torch.cat(cls_tkn, x[idx], dim = 1)
        x = self.backbone.norm(x)
        return x

    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.feature_forward(inputs)
        x = self.backbond.head(x)
        return x