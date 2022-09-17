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
                 **kwargs):

        super().__init__()
        if backbone_name is None:
            raise ValueError('backbone_name must be specified')
        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=class_num))
        self.reserve_rate = reserve_rate
        self.register_buffer('selection_layer', torch.tensor(selection_layer))
    
    def cls_append(self, x : torch.Tensor, **kwargs) -> torch.Tensor:
        B, N, C = x.shape
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.concat((cls_token, x), dim = 1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        return x

    def feature_forward(self, x : torch.Tensor, **kwargs) -> torch.Tensor:
        for n, block in enumerate(self.backbone.blocks):
            B, N, C = x.shape
            K = int(N * self.reserve_rate)
            layer = ((self.selection_layer == n).nonzero()).squeeze()

            norm = block.norm1(x)
            qkv = block.attn.qkv(norm).reshape(B, N, 3, block.attn.num_heads, C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
            
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            attn = block.attn.attn_drop(attn)

            if layer.numel() != 0:
                evidence = F.relu(attn)
                strength = (evidence + 1).sum(-1)
                uncertainty = (N / strength).mean(1)
                pass

            norm = (attn @ v).transpose(1, 2).reshape(B, N, C)
            norm = block.attn.proj(norm)
            norm = block.attn.proj_drop(norm)

            x = x + block.drop_path(norm)
            if layer.numel() != 0:
                cls_tkn = x[:, 0].unsqueeze(1)
                img_tkn = x[:, 1:]
                _, idx = uncertainty.topk(K, largest = True, sorted = True)
                _, stl = uncertainty.topk(N - K, largest = True, sorted = True)
                stl_tkn = img_tkn.gather(1, stl.unsqueeze(-1).expand(-1, -1, C)).sum(1).unsqueeze(1)
                img_tkn = img_tkn.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))
                x = torch.concat((cls_tkn, img_tkn, stl_tkn), dim = 1)
            x = x + block.drop_path(block.mlp(block.norm2(x)))
        return x

    def forward(self, x : torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.backbone.patch_embed(x)
        x = self.cls_append(x)
        x = self.feature_forward(x)
        x = self.backbone.head(x[:, 0])
        return x