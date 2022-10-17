from typing import TypeVar

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

T = TypeVar('T', bound = 'nn.Module')

class NotL2P(nn.Module):
    def __init__(self,
                 num_task      : int    = 10,
                 selection_size: int    = 5,
                 prompt_len     : int   = 5,
                 class_num      : int   = 100,
                 backbone_name  : str   = None,
                 lambd          : float = 0.5,
                 shared_prompts : int   = 2,
                 _batchwise_selection  : bool = True,
                 _diversed_selection   : bool = True,
                 _scale_prompts        : bool = True,
                 _unsim_penalty        : bool = True,
                 _scale_simmilarity    : bool = True,
                 _update_per_iter      : bool = False,
                 **kwargs):
        super().__init__()
        
        if backbone_name is None:
            raise ValueError('backbone_name must be specified')

        self.selection_size = selection_size
        self.prompt_len     = prompt_len
        self.lambd          = lambd
        self.shared_prompts = shared_prompts
        self.class_num      = class_num
        self._diversed_selection  = _diversed_selection
        self._batchwise_selection = _batchwise_selection
        self._scale_prompts       = _scale_prompts
        self._unsim_penalty       = _unsim_penalty
        self._scale_simmilarity   = _scale_simmilarity
        self._update_per_iter     = _update_per_iter

        self.num_task = num_task
        self.task_id = -1 # if _convert_train_task is not called, task will undefined
        self.tasks = []
        
        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=class_num))
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.head.weight.requires_grad = True
        self.backbone.head.bias.requires_grad = True

        self.prompt = nn.Parameter(torch.rand((num_task, selection_size - shared_prompts, prompt_len, self.backbone.num_features)),requires_grad=True)
        self.shared_prompt = nn.Parameter(torch.rand((shared_prompts, prompt_len, self.backbone.num_features)),requires_grad=True)

        self.register_buffer('simmilarity', torch.zeros(1))
        self.register_buffer('unsimmilarity', torch.zeros(1))
        self.register_buffer('mask', torch.zeros(class_num))
    
    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        
        x = self.backbone.patch_embed(inputs)
        B, N, D = x.shape
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        t = torch.cat((cls_token, x), dim=1)

        ep = self.prompt[self.task_id].repeat(B,1,1,1).view(B, -1, D)
        ep = ep + self.backbone.pos_embed[:,0].clone().expand((self.selection_size - self.shared_prompts) * self.prompt_len, -1)
        sp = self.shared_prompt.repeat(B,1,1).view(B, -1, D)
        sp = sp + self.backbone.pos_embed[:,0].clone().expand(self.shared_prompts * self.prompt_len, -1)

        x = self.backbone.pos_drop(t + self.backbone.pos_embed)
        x = torch.cat((x[:,0].unsqueeze(1), ep, sp, x[:,1:]), dim=1)

        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)

        x = x[:, 1:self.selection_size * self.prompt_len + 1].clone()
        x = x.mean(dim=1)
        x = self.backbone.head(x)

        if self.training:
            x = x + self.mask
        return x
    
    def loss_fn(self, output, target):
        B, C = output.size()
        return F.cross_entropy(output, target)

    def _convert_train_task(self, task : torch.Tensor, **kwargs):
        flag = -1
        for n, t in enumerate(self.tasks):
            if torch.equal(t, task):
                flag = n
                break
        if flag == -1:
            self.tasks.append(task)
            self.task_id = len(self.tasks) - 1
        else :
            self.task_id = flag

        self.mask += -torch.inf
        self.mask[task] = 0
        
        return torch.zeros(self.num_task)

    def train(self: T, mode: bool = True, **kwargs) -> T:
        ten = super().train(mode)
        self.backbone.eval()
        return ten
