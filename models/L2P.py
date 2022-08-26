import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.prompt import Prompt
from typing import TypeVar
T = TypeVar('T', bound = 'nn.Module')

class L2P(nn.Module):
    def __init__(self,
                 pool_size      : int,
                 selection_size : int,
                 dimention      : int,
                 prompt_len     : int,
                 class_num      : int,
                 backbone_name  : str,
                 *args, **kwargs):

        super(L2P, self).__init__()

        self.selection_size = selection_size
        self.prompt_len     = prompt_len
        self.class_num      = class_num
        self.dimention      = dimention
        
        self.backbone = timm.create_model(backbone_name, pretrained=True)
        self.add_module('backbone', self.backbone)
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(self.backbone.no_embed_class)

        self.prompt         = Prompt(pool_size, selection_size, prompt_len, dimention)
        self.simmilairty    = 0.0
        self.avgpool        = nn.AdaptiveAvgPool2d((1,dimention))

        self.past_class     = torch.zeros(class_num)

        self.classifier        = nn.Linear     (dimention, class_num)
        self.classifier.weight = nn.init.zeros_(self.classifier.weight)
        self.classifier.bias   = nn.init.zeros_(self.classifier.bias)

    def forward(self, inputs : torch.Tensor, *args, **kwargs):
        self.past_class = self.past_class.to(next(self.parameters()).device)

        x = self.backbone.patch_embed(inputs)
        x = self.backbone._pos_embed(x)
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
            x = x + self.past_class
        x = F.softmax(x, dim = -1)
        self.simmilairty = _simmilarity.sum() / x.size()[0]
        return x
        
    def loss_fn(self, output, target):
        return nn.CrossEntropyLoss(label_smoothing=0.1)(output, target) - 0.5 * self.simmilairty

    def accuracy(self, output, target):
        return (output.argmax(dim = 1) == target).sum()/ output.size()[0]

    def task_mask(self, mask : torch.Tensor):
        self.past_class += -torch.inf
        self.past_class[mask] = 0
        return self.past_class

    def train(self: T, mode: bool = True) -> T:
        ten = super().train(mode)
        self.backbone.eval()
        return ten

    def freeze_classifier(self, mask : torch.Tensor):
        for param in self.classifier.parameters():
            param.requires_grad = False
        return self