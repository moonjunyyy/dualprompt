from typing import TypeVar

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Prompt import Prompt

T = TypeVar('T', bound = 'nn.Module')

class InViTL2P(nn.Module):
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
        self._batchwise_selection = _batchwise_selection
        self._scale_prompts       = _scale_prompts
        self._unsim_penalty       = _unsim_penalty
        self._scale_simmilarity   = _scale_simmilarity
        self._update_per_iter     = _update_per_iter
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
            _diversed_selection  = _diversed_selection,
            _batchwise_selection = _batchwise_selection,
            _get_unsimmilarity   = _unsim_penalty)

        self.register_buffer('simmilarity', torch.zeros(1))
        self.register_buffer('unsimmilarity', torch.zeros(1))
        self.register_buffer('mask', torch.zeros(class_num))
    
    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:

        img = self.backbone.patch_embed(inputs)
        B, N, D = img.size()
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        token_appended = torch.cat((cls_token, img), dim=1)
        x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
        
        query = self.backbone.blocks(x)
        query = self.backbone.norm(query)
        query = query[:, 0]

        if self._unsim_penalty:
            simmilarity, unsimmilarity, prompts, _ = self.prompt(query)
        else:
            simmilarity, prompts = self.prompt(query)

        if self._scale_simmilarity:
            freq = F.normalize(self.prompt.frequency.reciprocal(), p=1, dim=-1)
            self.simmilarity       =  (simmilarity * freq.repeat(B, 1).gather(1, self.prompt.topk)).sum()
            if self._unsim_penalty:
                self.unsimmilarity = (unsimmilarity * freq.repeat(B, 1).gather(1, self.prompt.nonk)).sum()
        else:
            self.simmilarity       =   simmilarity.sum()
            if self._unsim_penalty:
                self.unsimmilarity = unsimmilarity.sum()
            
        if self._scale_prompts :
            scale = ((simmilarity - 1).detach() * self.tau).exp()
            prompts = (prompts * scale.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.prompt_len, D)).contiguous().view(B, self.selection_size * self.prompt_len, D)
        else :
            prompts = prompts.contiguous().view(B, self.selection_size * self.prompt_len, D)

        prompts = prompts + self.backbone.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)

        img = img + self.backbone.pos_embed[:,1:].clone()
        if self.training:
            sel = torch.rand((B, N),device=x.device).argsort(dim=1)
            img = torch.concat((
                img.gather(1, sel[:, :int(N*0.5)].unsqueeze(-1).expand(-1, -1, D)),
                img.gather(1, sel[:, int(N*0.5):].unsqueeze(-1).expand(-1, -1, D)))
                , dim = 0)
            x = torch.cat((x[:,:1].repeat(2,1,1), prompts.repeat(2,1,1), img), dim=1)
        else:
            x = torch.cat((x[:,:1], prompts, img), dim=1)

        x = self.backbone.pos_drop(x)

        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        x = x[:, 1:self.selection_size*self.prompt_len+1].clone()

        if self.training:
            self.imgloss = x[:B].clone().transpose(1,2) @ x[B:].clone()
            self.imgloss = self.imgloss / self.imgloss.max()
            self.imgloss = -(self.imgloss.diagonal(dim1=1, dim2=2).exp().sum() / self.imgloss.exp().sum()).log()        
            x = x.mean(dim=1)
        else:
            x = x.mean(dim=1)
        x = self.backbone.head(x)
        if self.training:
            x = (x[:B].clone() + x[B:].clone())/2 
            x = x + self.mask
        return x
    
    def loss_fn(self, output, target):
        B, C = output.size()
        if self._unsim_penalty:
            return F.cross_entropy(output, target) + self.lambd * self.simmilarity - self.xi * self.unsimmilarity
        else :
            if self.training:
                return F.cross_entropy(output, target) + self.lambd * self.simmilarity + self.xi * self.imgloss
            else:
                return F.cross_entropy(output, target) + self.lambd * self.simmilarity
                
    def convert_train_task(self, task : torch.Tensor, **kwargs):
        self.mask += -torch.inf
        self.mask[task.to(self.mask.device)] = 0
        return self.prompt.update()

    def get_count(self):
        return self.prompt.update()
        
    def train(self: T, mode: bool = True, **kwargs) -> T:
        ten = super().train(mode)
        self.backbone.eval()
        return ten
