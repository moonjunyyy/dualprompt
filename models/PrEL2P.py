from typing import TypeVar

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
T = TypeVar('T', bound = 'nn.Module')

class PrEL2P(nn.Module):
    def __init__(self,
                 vit_name        : str   = None,
                 class_num       : int   = 100,
                 reserve_rate    : float = 0.7,
                 selection_layer : tuple = (3,),
                 _learnable_pos_emb : bool = False,
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
        
        self.pos_embed = nn.Parameter(self.backbone.pos_embed.clone().detach(), requires_grad=_learnable_pos_emb)
    
    def cls_append(self, x : torch.Tensor, **kwargs) -> torch.Tensor:
        B, N, C = x.size()
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        return x

    def feature_forward(self, x : torch.Tensor, reserved_prompts = 0, **kwargs) -> torch.Tensor:
        B, L, D = x.size()
        for n, block in enumerate(self.backbone.blocks):
            x = x + block.drop_path(block.attn(block.norm1(x)))
            layer = ((self.selection_layer == n).nonzero()).squeeze()
            if layer.numel() != 0:
                cls_tkn = x[:, 0].unsqueeze(1)
                pmt_tkn = x[:, 1:reserved_prompts] if reserved_prompts != 0 else None
                img_tkn = x[:, 1+reserved_prompts:]
                K   = int((L - reserved_prompts) * self.reserve_rate)
                sim = F.cosine_similarity(img_tkn, cls_tkn, dim = -1)
                sim, idx = sim.topk(K, dim = -1, sorted=False)
                img_tkn  = img_tkn.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
                rst_tkn  = (x[:,1:].sum(dim = -2, keepdim = True) - img_tkn.sum(dim = -2, keepdim = True))
                if pmt_tkn is None:
                    x    = torch.cat([cls_tkn, img_tkn, rst_tkn], dim = 1)
                else:
                    x    = torch.cat([cls_tkn, pmt_tkn, img_tkn, rst_tkn], dim = 1)
            x = x + block.drop_path(block.mlp(block.norm2(x)))
        x = self.backbone.norm(x)
        return x

    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.backbone.patch_embed(inputs)
        x = self.cls_append(x)
        x = self.feature_forward(x)
        x = self.backbond.head(x)
        return x