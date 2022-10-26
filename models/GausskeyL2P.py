from cmath import isinf
from typing import TypeVar

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Prompt import Prompt

T = TypeVar('T', bound = 'nn.Module')

class GausskeyL2P(nn.Module):
    def __init__(self,
                 pool_size      : int   = 10,
                 selection_size : int   = 5,
                 prompt_len     : int   = 5,
                 class_num      : int   = 100,
                 psudo_sapmle   : int   = 10,
                 backbone_name  : str   = None,
                 lambd          : float = 0.5,
                 zetta          : float = 0.1,
                 xi             : float = 0.1,
                 _batchwise_selection  : bool = True,
                 _diversed_selection   : bool = True,
                 _update_per_iter      : bool = False,
                 **kwargs):

        super().__init__()
        
        if backbone_name is None:
            raise ValueError('backbone_name must be specified')
        if pool_size < selection_size:
            raise ValueError('pool_size must be larger than selection_size')

        self.pool_size      = pool_size
        self.prompt_len     = prompt_len
        self.selection_size = selection_size
        self.psudo_sapmle   = psudo_sapmle
        self.lambd = lambd
        self.zetta = zetta
        self.xi    = xi
        self._batchwise_selection = _batchwise_selection
        self._update_per_iter     = _update_per_iter
        self.class_num      = class_num

        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=class_num))
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.head.weight.requires_grad = True
        self.backbone.head.bias.requires_grad = True
        
        self.prompt   = nn.Parameter(torch.randn(pool_size, prompt_len, self.backbone.num_features), requires_grad=True)
        
        self.mean     = nn.Parameter(torch.randn(self.pool_size, self.backbone.num_features), requires_grad=True)
        self.variance = nn.Parameter(torch.eye(self.backbone.num_features).repeat(self.pool_size,1,1), requires_grad = True)

        # self.register_buffer('mean',     torch.rand(self.pool_size, self.backbone.num_features, requires_grad=True))
        # self.register_buffer('variance', torch.eye(self.backbone.num_features, requires_grad=True).repeat(self.pool_size, 1, 1))

        self.prior_mean = self.mean.clone().detach()
        self.register_buffer('frequency', torch.ones(self.pool_size))
        self.register_buffer('sample', torch.zeros((pool_size, class_num)))
        self.register_buffer('mask',          torch.zeros(class_num))

    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:

        self.distribution = torch.distributions.MultivariateNormal(self.mean, scale_tril=self.variance)
        x = self.backbone.patch_embed(inputs)

        B, N, D = x.size()
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        t = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(t + self.backbone.pos_embed)
        q = self.backbone.blocks(x)
        q = self.backbone.norm(q)[:, 0].clone()
        # B, P
        # prob = self.distribution.log_prob(q.unsqueeze(1))
        prob =  self.distribution.log_prob(q.unsqueeze(1))
        topk = prob * F.normalize(self.frequency, p=1, dim=-1) #Diversed Selection
        _ ,topk = topk.topk(self.selection_size, dim=-1, largest=True, sorted=True)
        self.frequency += torch.bincount(topk[:,:3].contiguous().view(-1), minlength = self.pool_size)
        p = self.prompt.repeat(B, 1, 1, 1).gather(1, topk.unsqueeze(-1).unsqueeze(-1).expand(B, -1, self.prompt_len, self.backbone.num_features))
        
        # B, P, D
        # N, P, D
        # B, P, N, D
        psudo_sample = self.distribution.sample((self.psudo_sapmle,)).permute(1, 0, 2)[topk[:,:3],:,:]
        # psudo_sample = torch.cat((psudo_sample, q.unsqueeze(1).unsqueeze(1).expand(-1, 3,-1,-1)), dim=2)
        # B, P, D
        self.mean_loss = (self.mean[topk[:,:3]].unsqueeze(2) - psudo_sample).pow(2).mean() * self.lambd + (self.mean[topk[:,:3]] - q.unsqueeze(1)).pow(2).mean()
        self.var_loss  = - self.distribution.log_prob(psudo_sample.unsqueeze(3)).mean() * self.zetta + self.distribution.log_prob(q.unsqueeze(1)).mean()

        p  =  p.contiguous().view(B, self.selection_size * self.prompt_len, D)
        p  =  p + self.backbone.pos_embed[:,0].expand(self.selection_size * self.prompt_len, -1)

        x = self.backbone.pos_drop(t + self.backbone.pos_embed)
        sx = torch.cat((x[:,0].unsqueeze(1),  p, x[:,1:]), dim=1)

        sx = self.backbone.blocks(sx)
        sx = self.backbone.norm(sx)
        sx = sx[:, 1:self.selection_size * self.prompt_len + 1].clone()

        sx = sx.mean(dim=1)
        x = self.backbone.head(sx)

        if self._update_per_iter:
            self.prompt.update()

        if self.training:
            x = x + self.mask
        return x
    
    def loss_fn(self, output, target):
        B, C = output.size()
        return F.cross_entropy(output, target) + self.mean_loss + self.var_loss + self.xi * self.variance.pow(2).mean()

    def _convert_train_task(self, task : torch.Tensor, **kwargs):
        self.mask += -torch.inf
        self.mask[task.to(self.mask.device)] = 0
        return self.frequency

    def train(self: T, mode: bool = True, **kwargs) -> T:
        ten = super().train(mode)
        self.backbone.eval()
        return ten
