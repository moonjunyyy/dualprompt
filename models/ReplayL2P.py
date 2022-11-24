from typing import TypeVar

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from layers.Prompt import Prompt

T = TypeVar('T', bound = 'nn.Module')

class ReplayL2P(nn.Module):
    def __init__(self,
                 pool_size      : int   = 10,
                 selection_size : int   = 5,
                 prompt_len     : int   = 5,
                 class_num      : int   = 100,
                 backbone_name  : str   = None,
                 lambd          : float = 0.5,
                 tau            : float = 0.5,
                 xi             : float = 0.1,
                 _batchwise_selection  : bool = False,
                 _diversed_selection   : bool = True,
                 _scale_prompts        : bool = False,
                 _unsim_penalty        : bool = False,
                 _scale_simmilarity    : bool = False,
                 _update_per_iter      : bool = False,
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
        self._batchwise_selection = False
        self._scale_prompts       = False
        self._unsim_penalty       = False
        self._scale_simmilarity   = False
        self._update_per_iter     = False
        self.class_num            = class_num

        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=class_num))
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.head.weight.requires_grad = True
        self.backbone.head.bias.requires_grad   = True

        self.prompt = Prompt(
            pool_size,
            selection_size,
            prompt_len,
            self.backbone.num_features,
            _diversed_selection  = False,
            _batchwise_selection = False,
            _get_unsimmilarity   = False)

        self.image_buffer = nn.parameter.Parameter(torch.rand(class_num, 196, self.backbone.num_features), requires_grad=True)

        self.register_buffer('simmilarity', torch.zeros(1))
        self.register_buffer('unsimmilarity', torch.zeros(1))
        self.register_buffer('mask', torch.zeros(class_num))
        self.register_buffer('replay_mask', torch.zeros(class_num, dtype=torch.bool))
    
    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        B,_,_,_ = inputs.size()
        
        if self.training:
            
            self.image_buffer_replay(self.replay_mask, B)

        x = self.backbone.patch_embed(inputs)
        B, N, D = x.size()

        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        token_appended = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)

        query = self.backbone.blocks(x)
        query = self.backbone.norm(query)[:, 0].clone()
        simmilarity, prompts = self.prompt(query)
        self.simmilarity = simmilarity.sum()

        prompts = prompts.contiguous().view(B, self.selection_size * self.prompt_len, D)
        prompts = prompts + self.backbone.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)

        x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
        x = torch.cat((x[:,0].unsqueeze(1), prompts, x[:,1:]), dim=1)

        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)

        x = x[:, 1:self.selection_size * self.prompt_len + 1].clone()
        x = x.mean(dim=1)
        x = self.backbone.head(x)

        if self.training:
            x = x + self.mask

        return x

    def image_buffer_replay(self, mask, batch_size, **kwargs) -> None:

        classes = torch.arange(0, self.class_num, dtype=torch.long, device=self.mask.device)
        #classes = classes[mask]
        class_num = classes.size(0)

        if dist.is_available():
            rank = dist.get_rank()
            ngpu = dist.get_world_size()
        else:
            rank = 0
            ngpu = 1

        iter   =   class_num // (batch_size  * ngpu) + 1
        offset =  (class_num // ngpu) * rank
        
        for i in range(iter):
            if offset + i * ngpu * batch_size >= class_num:
                break
            if  offset + (i + 1) * batch_size >= class_num:
                image = self.image_buffer[classes[offset + i * batch_size * ngpu : (class_num // ngpu) * (rank + 1)]]
            else:
                image = self.image_buffer[classes[offset + i * batch_size * ngpu : offset + (i + 1) * batch_size * ngpu]]
            B, N, D = image.size()

            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            token_appended = torch.cat((cls_token, image), dim=1)
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)

            query = self.backbone.blocks(x)
            query = self.backbone.norm(query)[:, 0].clone()

            simmilarity, prompts = self.prompt(query)
            prompts = prompts.contiguous().view(B, self.selection_size * self.prompt_len, D)
            prompts = prompts + self.backbone.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
            x = torch.cat((x[:,0].unsqueeze(1), prompts, x[:,1:]), dim=1)

            x = self.backbone.blocks(x)
            x = self.backbone.norm(x)

            x = x[:, 1:self.selection_size * self.prompt_len + 1].clone()
            x = x.mean(dim=1)
            x = self.backbone.head(x)

            lable = classes[rank + i * batch_size * ngpu : rank + i * batch_size * ngpu + B]
            (F.cross_entropy(x, lable) * 0.01).backward()

    def loss_fn(self, output, target):
        B, C = output.size()
        return F.cross_entropy(output, target) + self.lambd * self.simmilarity

    def convert_train_task(self, task : torch.Tensor, **kwargs):
        self.mask += -torch.inf
        self.mask[task.to(self.mask.device)] = 0

        self.replay_mask[self.mask.eq(0)] = True
        return

    def get_count(self):
        return self.prompt.update()

    def train(self: T, mode: bool = True, **kwargs) -> T:
        ten = super().train(mode)
        self.backbone.eval()
        return ten
