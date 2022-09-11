from typing import TypeVar

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

T = TypeVar('T', bound = 'nn.Module')

class EViT(nn.Module):
    def __init__(self,
                 vit_name        : str   = None,
                 class_num       : int   = 100,
                 reserve_rate    : float = 0.7,
                 selection_layer : tuple = (3,),
                 **kwargs):

        super().__init__()
        
        if vit_name is None:
            raise ValueError('vit_name must be specified')
        self.reserve_rate = reserve_rate
        self.register_buffer('selection_layer', torch.tensor(selection_layer))

        self.add_module('backbone', timm.create_model(vit_name, pretrained=True, num_classes=class_num))
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.head.weight.requires_grad = True
        self.backbone.head.bias.requires_grad = True
    
    def cls_append(self, x : torch.Tensor, **kwargs) -> torch.Tensor:
        B, N, C = x.size()
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        return x

    def feature_forward(self, x : torch.Tensor, **kwargs) -> torch.Tensor:
        B, L, D = x.size()
        for n, block in enumerate(self.backbone.blocks):
            x = block(x)
            layer = ((self.selection_layer == n).nonzero()).squeeze()
            if layer.numel() != 0:
                cls_tkn = x[:,0].unsqueeze(1)
                img_tkn = x[:,1:]
                sim = F.cosine_similarity(img_tkn, cls_tkn, dim = -1)
                sim, idx = sim.topk(int(L * self.reserve_rate), dim = -1, sorted=False)
                img_tkn = img_tkn.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
                x = torch.cat([cls_tkn, img_tkn], dim = 1)
        x = self.backbone.norm(x)
        return x

    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.backbone.patch_embed(inputs)
        x = self.cls_append(x)
        x = self.feature_forward(x)
        x = self.backbond.head(x)
        return x