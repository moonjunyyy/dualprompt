from turtle import forward
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.prompt import Prompt
from typing import TypeVar
T = TypeVar('T', bound = 'nn.Module')

class L2P(nn.Module):
    def __init__(self,
                 pool_size      : int   = 10,
                 selection_size : int   = 5,
                 prompt_len     : int   = 5,
                 class_num      : int   = 100,
                 backbone_name  : str   = None,
                 lambd          : float = 0.1,
                 batchwise_selection : bool = False,
                 **kwargs):
        super().__init__()
        
        if backbone_name is None:
            raise ValueError('backbone_name must be specified')
        if pool_size < selection_size:
            raise ValueError('pool_size must be larger than selection_size')

        self.prompt_len = prompt_len
        self.selection_size = selection_size
        self.lambd = lambd
        self.batchwise_selection = batchwise_selection

        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=class_num))
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.head.requires_grad = True
        self.prompt = Prompt(pool_size, selection_size, prompt_len, self.backbone.num_features, batchwise_selection = batchwise_selection)
        self.register_buffer('simmilarity', torch.zeros(1))
        self.register_buffer('mask', torch.zeros(class_num))

        self.avgpool = nn.AdaptiveAvgPool2d((1, self.backbone.num_features))

    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        
        x = self.backbone.patch_embed(inputs)
        B, N, D = x.size()

        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)

        q = self.backbone.blocks(x)
        q = self.backbone.norm(q)

        s, p = self.prompt(q[:, 0])
        self.simmilarity = s.mean()

        p = p.reshape(B, self.selection_size * self.prompt_len, D)
        p = p + self.backbone.pos_embed[:,0].expand(B, self.selection_size * self.prompt_len, -1)
        x = torch.cat((p , x), dim=1)

        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        x = x[:, :self.selection_size * self.prompt_len]
        x = self.avgpool(x).squeeze()
        x = self.backbone.head(x)
        if self.training:
            x = x + self.mask
            x = x.softmax(dim=-1)

        return x
    
    def loss_fn(self, output, target):
        return F.cross_entropy(output, target) + self.lambd * self.simmilarity

    def _convert_train_task(self, task : torch.Tensor, **kwargs):
        self.mask += -torch.inf
        self.mask[task] = 0
        return self.mask

    def train(self: T, mode: bool = True, **kwargs) -> T:
        ten = super().train(mode)
        self.backbone.eval()
        return ten