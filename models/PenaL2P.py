from typing import TypeVar

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.prompt import Prompt

T = TypeVar('T', bound = 'nn.Module')

class PenaL2P(nn.Module):
    def __init__(self,
                 pool_size      : int   = 10,
                 selection_size : int   = 5,
                 prompt_len     : int   = 5,
                 class_num      : int   = 100,
                 backbone_name  : str   = None,
                 lambd          : float = 0.5,
                 tau            : float = 0.5,
                 xi             : float = 0.1,
                 _batchwise_selection  : bool = True,
                 _diversed_selection   : bool = True,
                 _scale_prompts        : bool = True,
                 _unsim_penalty        : bool = True,
                 _scale_simmilarity    : bool = True,
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
        self._batchwise_selection = _batchwise_selection
        self._scale_prompts       = _scale_prompts
        self._unsim_penalty       = _unsim_penalty
        self._scale_simmilarity   = _scale_simmilarity
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
            _get_unsimmilarity   = _unsim_penalty)

        self.register_buffer('simmilarity', torch.zeros(1))
        self.register_buffer('unsimmilarity', torch.zeros(1))
        self.register_buffer('mask', torch.zeros(class_num))
        
    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.backbone.patch_embed(inputs)

        B, N, D = x.size()
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        t = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(t + self.backbone.pos_embed)

        q = self.backbone.blocks(x)
        q = self.backbone.norm(q)[:, 0].clone()

        if self._unsim_penalty:
            s, us, p = self.prompt(q)
        else:
            s, p = self.prompt(q)

        if self._scale_simmilarity:
            freq = F.normalize(self.prompt.frequency.reciprocal(), p=1, dim=-1)
            self.simmilarity   =  (s * freq.repeat(B, 1).gather(1, self.prompt.topk)).sum()
            self.unsimmilarity = (us * freq.repeat(B, 1).gather(1, self.prompt.nonk)).sum()
        else:
            self.simmilarity   =  s.sum()
            self.unsimmilarity = us.sum()

            
        if self._scale_prompts :
            scale = ((s - 1) * self.tau).exp()
            p = (p * scale.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.prompt_len, D)).contiguous().view(B, self.selection_size * self.prompt_len, D)
        else :
            p = p.contiguous().view(B, self.selection_size * self.prompt_len, D)
        p = p + self.backbone.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)

        x = self.backbone.pos_drop(t + self.backbone.pos_embed)
        x = torch.cat((x[:,0].unsqueeze(1), p, x[:,1:]), dim=1)

        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)

        x = x[:, 1:self.selection_size * self.prompt_len + 1].clone()
        x = x.mean(dim=1)
        x = self.backbone.head(x)
        self.prompt.update()
        if self.training:
            x = x + self.mask
        return x
    
    def loss_fn(self, output, target):
        B, C = output.size()
        if self._unsim_penalty:
            return F.cross_entropy(output, target) - self.lambd * self.simmilarity + self.xi * self.unsimmilarity
        else :
            return F.cross_entropy(output, target) - self.lambd * self.simmilarity

    def _convert_train_task(self, task : torch.Tensor, **kwargs):
        self.mask += -torch.inf
        self.mask[task.to(self.mask.device)] = 0
        return self.prompt.update()

    def train(self: T, mode: bool = True, **kwargs) -> T:
        ten = super().train(mode)
        self.backbone.eval()
        return ten