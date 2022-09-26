from typing import TypeVar

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.prompt import Prompt

T = TypeVar('T', bound = 'nn.Module')

class GL2P(nn.Module):
    def __init__(self,
                 pool_size      : int   = 10,
                 selection_size : int   = 5,
                 prompt_len     : int   = 5,
                 class_num      : int   = 100,
                 backbone_name  : str   = None,
                 lambd          : float = 0.5,
                 _cls_at_front         : bool = False,
                 _batchwise_selection  : bool = True,
                 _mixed_prompt_order   : bool = False,
                 _mixed_prompt_token   : bool = False,
                 _learnable_pos_emb    : bool = False,
                 _sub_from_prompts     : bool = False,
                 _Hadamard_Mul         : bool = False,  
                 **kwargs):

        super().__init__()
        
        if backbone_name is None:
            raise ValueError('backbone_name must be specified')
        if pool_size < selection_size:
            raise ValueError('pool_size must be larger than selection_size')

        self.prompt_len = prompt_len
        self.selection_size = selection_size
        self.lambd = lambd
        self._batchwise_selection = _batchwise_selection
        self.class_num = class_num
        self._cls_at_front = _cls_at_front
        self._sub_from_prompts = _sub_from_prompts
        self._Hadamard_Mul = _Hadamard_Mul

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
            _batchwise_selection = _batchwise_selection,
            _mixed_prompt_order = _mixed_prompt_order,
            _mixed_prompt_token = _mixed_prompt_token)

        self.pos_embed = self.backbone.pos_embed.clone().detach()
        self.pos_embed = nn.Parameter(self.pos_embed, requires_grad=_learnable_pos_emb)
        
        self.register_buffer('simmilarity', torch.zeros(1))
        self.register_buffer('mask', torch.zeros(class_num))
        self.register_buffer('generalizer', torch.zeros(1, prompt_len, self.backbone.num_features))
        
    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        
        with torch.no_grad():
            if self._Hadamard_Mul:
                # P L C
                if self.generalizer.sum() == 0:
                    self.generalizer = self.generalizer + 1
                for n in range(self.prompt.pool_size):
                    self.generalizer = self.generalizer * F.normalize(self.prompt.prompts.data[n])
                self.generalizer = F.normalize(self.generalizer)
                if self._sub_from_prompts:
                    self.prompt.prompts.data = self.prompt.prompts.data - self.generalizer
            else:
                self.generalizer = (self.prompt.prompts.data + self.generalizer).mean(0).unsqueeze(0)
                if self._sub_from_prompts:
                    self.prompt.prompts.data = self.prompt.prompts.data - self.generalizer

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
        p = torch.concat((self.generalizer.expand(B, -1, -1), p), dim = 1)
        p = p + self.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len + self.prompt_len, -1)

        x = self.backbone.pos_drop(t + self.pos_embed)
        if self._cls_at_front:
            x = torch.cat((x[:,0].unsqueeze(1), p, x[:,1:]), dim=1)
        else :
            x = torch.cat((p, x), dim=1)

        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)

        if self._cls_at_front:
            x = x[:, 1:(self.selection_size + 1) * self.prompt_len + 1].clone()
        else :
            x = x[:,  :(self.selection_size + 1) * self.prompt_len].clone()
        x = x.mean(dim=1)
        x = self.backbone.head(x)

        if self.training:
            x = x + self.mask
        return x
    
    def loss_fn(self, output, target):
        B, C = output.size() 
        return F.cross_entropy(output, target) - self.lambd * self.simmilarity

    def _convert_train_task(self, task : torch.Tensor, **kwargs):
        with torch.no_grad():
            self.mask += -torch.inf
            self.mask[task.to(self.mask.device)] = 0
        return self.prompt.update()

    def train(self: T, mode: bool = True, **kwargs) -> T:
        ten = super().train(mode)
        self.backbone.eval()
        return ten
