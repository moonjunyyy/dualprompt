from typing import TypeVar

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.prompt import Prompt
T = TypeVar('T', bound = 'nn.Module')

class PrEL2P(nn.Module):
    def __init__(self,
                 backbone_name   : str   = None,
                 class_num       : int   = 100,
                 pool_size       : int   = 10,
                 selection_size  : int   = 4,
                 reserve_rate    : float = 0.7,
                 selection_layer : tuple = (3,),
                 lambd           : float = 0.1,
                 _learnable_pos_emb : bool = False,
                 **kwargs):

        super().__init__()
        if backbone_name is None:
            raise ValueError('vit_name must be specified')
        
        self.reserve_rate = reserve_rate
        self.selection_size = selection_size
        self.register_buffer('selection_layer', torch.tensor(selection_layer))
        
        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=class_num))
        num_patches = self.backbone.patch_embed.num_patches
        self.num_stale   = ((num_patches - int(num_patches * self.reserve_rate)) // selection_size) * selection_size
        self.num_reserve = num_patches - self.num_stale
        self.lambd = lambd

        self.prompts = Prompt(pool_size, selection_size, len(self.selection_layer) * self.num_stale // selection_size, self.backbone.num_features)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.head.weight.requires_grad = True
        self.backbone.head.bias.requires_grad = True

        self.register_buffer('simmilarity', torch.zeros(1))
        self.register_buffer('mask', torch.zeros(class_num))
        self.pos_embed = nn.Parameter(self.backbone.pos_embed.clone().detach(), requires_grad=_learnable_pos_emb)
    
    def feature_forward(self, x : torch.Tensor, prompts : torch.Tensor, **kwargs) -> torch.Tensor:
        B, N, C = x.size()
        prompts = prompts.reshape(B, self.selection_size, len(self.selection_layer), -1, C)
        prompts = prompts.permute(2,0,1,3,4).reshape(len(self.selection_layer), B, -1, C)
        for n, block in enumerate(self.backbone.blocks):
            r = x
            x = block.norm1(x)

            msa   = block.attn
            qkv = msa.qkv(x).reshape(B, N, 3, msa.num_heads, C // msa.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

            attn = (q @ k.transpose(-2, -1)) * msa.scale
            attn = attn.softmax(dim=-1)
            attn = msa.attn_drop(attn)

            layer = ((self.selection_layer == n).nonzero()).squeeze()
            if layer.numel() != 0:
                importance = attn[:, :, 0, :].mean(dim = 1).clone()

            x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
            x = msa.proj(x)
            x = msa.proj_drop(x)
            x = r + block.drop_path(x)

            layer = ((self.selection_layer == n).nonzero()).squeeze()
            if layer.numel() != 0:
                i, idx = importance[:,1:].clone().topk(self.num_stale, dim = -1, sorted=False, largest=False)
                x[:,idx + 1] = prompts[layer].squeeze() + self.pos_embed.expand(B,-1,-1)[:,idx + 1]
            x = x + block.drop_path(block.mlp(block.norm2(x)))
        x = self.backbone.norm(x)
        return x

    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        self.simmilarity = torch.zeros(1, device=self.simmilarity.device)
        x = self.backbone.patch_embed(inputs)
        B, N, D = x.size()
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        t = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(t + self.backbone.pos_embed)

        q = self.backbone.blocks(x)
        q = self.backbone.norm(q)[:, 0].clone()
        s, p = self.prompts(q)
        self.simmilarity = s.sum()
        x = self.feature_forward(x, p)
        x = self.backbone.head(x[:,0].clone())

        if self.training:
            x = x + self.mask
        return x

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target) - self.lambd * self.simmilarity

    def _convert_train_task(self, task : torch.Tensor, **kwargs):
        self.mask += -torch.inf
        self.mask[task.to(self.mask.device)] = 0
        updates = self.prompts.update()
        return torch.tensor(updates)