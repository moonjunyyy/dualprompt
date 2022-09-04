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
                 **kwargs):

        super(L2P, self).__init__()
        if backbone_name is None:
            raise ValueError('backbone_name must be specified')

        self.selection_size = selection_size
        self.prompt_len     = prompt_len
        self.class_num      = class_num
        
        self.backbone = timm.create_model(backbone_name, pretrained=True)
        self.add_module('backbone', self.backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.dimention      = self.backbone.embed_dim

        self.prompt         = Prompt(pool_size, selection_size, prompt_len, self.dimention)
        self.simmilairty    = 0.0
        self.avgpool        = nn.AdaptiveAvgPool2d((1, self.dimention))

        self.past_class     = torch.zeros(class_num)

        self.classifier        = nn.Linear     (self.dimention, class_num)
        self.classifier.weight = nn.init.zeros_(self.classifier.weight)
        self.classifier.bias   = nn.init.zeros_(self.classifier.bias)

    def forward(self, inputs : torch.Tensor, **kwargs):
        self.to(inputs.device)
        x = self.backbone.patch_embed(inputs)
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        q = self.backbone.blocks(x)
        q = self.backbone.norm(q)[:, 0, :].clone()

        _simmilarity, prompts = self.prompt(q)

        prompts = prompts.contiguous().view(x.size()[0], -1, self.dimention)
        prompts = prompts + self.backbone.pos_embed[:,0,:].expand(prompts.size()[0], prompts.size()[1], -1)
        
        x = torch.concat([prompts, x], dim = 1)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)

        x = x[:, :self.selection_size * self.prompt_len, :].clone()
        x = self.avgpool(x).squeeze()
        x = self.classifier(x)
        if self.training:
            x = x + self.past_class.to(x.device)
        self.simmilairty = _simmilarity.sum() / x.size()[0]
        return x

    def loss_fn(self, output, target, **kwargs):
        return F.cross_entropy(output, target) - 0.5 * self.simmilairty

    def _convert_train_task(self, task : torch.Tensor, **kwargs):
        task = task.to(self.past_class.device)
        self.past_class += -torch.inf
        self.past_class[task] = 0
        return self.past_class

    def to(self, device : torch.device, **kwargs):
        for param in self.backbone.parameters():
            param.to(device)
        self.prompt = self.prompt.to(device)
        self.avgpool = self.avgpool.to(device)
        self.past_class = self.past_class.to(device)
        self.classifier = self.classifier.to(device)
        return self
        
    def train(self: T, mode: bool = True, **kwargs) -> T:
        ten = super().train(mode)
        self.backbone.eval()
        return ten
        