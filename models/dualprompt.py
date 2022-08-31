import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.prompt import Prompt
from models.L2P import L2P

class DualPrompt(L2P):
    def __init__(self,
                 dimention      : int   = None,
                 pos_g_prompt   : tuple = (),
                 len_g_prompt   : int   = 5,
                 pos_e_prompt   : tuple = (),
                 len_e_prompt   : int   = 20,
                 prompt_func    : str   = None,
                 task_num       : int   = 10,
                 class_num      : int   = 100,
                 backbone_name  : str   = None,
                 device         : torch.device = torch.device('cpu'),
                 **kwargs):
        super(DualPrompt, self).__init__(dimention= dimention,
                                         class_num= class_num,
                                         backbone_name= backbone_name,
                                         device= device, **kwargs)
                                         
        del(self.prompt)
        del(self.avgpool) 
        del(self.selection_size)
        del(self.prompt_len)

        if len(pos_e_prompt) == 0 and len(pos_g_prompt) == 0:
            raise ValueError('one of the pos_g_prompt or pos_e_prompt must be specified')

        self.pos_g_prompt = pos_g_prompt
        self.pos_e_prompt = pos_e_prompt
        self.len_g_prompt = len_g_prompt
        self.len_e_prompt = len_e_prompt

        if   prompt_func == 'prompt_tuning':
            self.prompt_func = self.prompt_tuning
            if len(pos_g_prompt) == 0:    
                self.g_prompt = None
            else:
                self.g_prompt = Prompt(1,        1, len(pos_g_prompt) * len_g_prompt, self.dimention, device=device)
            if len(pos_e_prompt) == 0:    
                self.e_prompt = None
            else:
                self.e_prompt = Prompt(task_num, 1, len(pos_e_prompt) * len_e_prompt, self.dimention, device=device)

        elif prompt_func == 'prefix_tuning':
            self.prompt_func = self.prefix_tuning
            if len(pos_g_prompt) == 0:    
                self.g_prompt = None
            else:
                self.g_prompt = Prompt(1,        1, len(pos_g_prompt) * 2 * len_g_prompt, self.dimention, device=device)
            if len(pos_e_prompt) == 0:    
                self.e_prompt = None
            else:
                self.e_prompt = Prompt(task_num, 1, len(pos_e_prompt) * 2 * len_e_prompt, self.dimention, device=device)
        else: raise ValueError('Unknown prompt_func: {}'.format(prompt_func))

        self.task_num = task_num
        self.task_id  = 0

    def forward(self, inputs : torch.Tensor, **kwargs):

        x = self.backbone.patch_embed(inputs)
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        q = self.backbone.blocks(x)
        q = self.backbone.norm(q)[:, 0, :].clone()
    
        if self.g_prompt is not None:
            g_prompt = self.g_prompt.prompt.clone()
            g_prompt = g_prompt.squeeze().reshape(-1, self.len_g_prompt, self.dimention)
            g_prompt = g_prompt.unsqueeze(0).repeat(x.size(0), 1, 1, 1)
        else:
            g_prompt = None
        
        if self.e_prompt is not None:
            if self.training:
                _simmilarity = F.cosine_similarity(q, self.e_prompt.key[self.task_id, :].clone(), dim=-1)
                e_prompt = self.e_prompt.prompt[self.task_id, :, :].clone()
                e_prompt = e_prompt.squeeze().reshape(-1, self.len_e_prompt, self.dimention)
                e_prompt = e_prompt.unsqueeze(0).repeat(x.size(0), 1, 1, 1)
            else:
                _simmilarity, e_prompt = self.e_prompt(q)
                e_prompt = e_prompt.squeeze().reshape(-1, self.len_e_prompt, self.dimention)
                e_prompt = e_prompt.unsqueeze(0).repeat(x.size(0), 1, 1, 1)
        else:
            e_prompt = None
            _simmilarity = torch.tensor([0.0],device=x.device)
            
        x = self.prompt_func(x, g_prompt, e_prompt)
        
        x = x[:, 0, :].clone()
        x = self.classifier(x)

        if self.training:
            x = x + self.past_class.to(x.device)
        if not self.training:
            x = F.softmax(x, dim = -1)

        self.simmilairty = _simmilarity.sum() / x.size()[0]
        return x
    
    def prompt_tuning(self, x : torch.Tensor, g_prompt : torch.Tensor, e_prompt : torch.Tensor, **kwargs):
        for n, block in enumerate(self.backbone.blocks):
            if n in self.pos_g_prompt:
                idx = self.pos_g_prompt.index(n)
                x = torch.cat((x, g_prompt[:, idx, :, :].clone()), dim = 1)
            if n in self.pos_e_prompt:
                idx = self.pos_e_prompt.index(n)
                x = torch.cat((x, e_prompt[:, idx, :, :].clone()), dim = 1)
            x = block(x)
        return x

    def prefix_tuning(self, x : torch.Tensor, g_prompt : torch.Tensor, e_prompt : torch.Tensor, **kwargs):

        for n, block in enumerate(self.backbone.blocks):
            r  = x
            x  = block.norm1(x)
            xk = x
            xv = x
            if n in self.pos_g_prompt:
                idx = self.pos_g_prompt.index(n)
                xk = torch.cat((xk, g_prompt[:, idx * 2 + 0, :, :].clone()), dim = 1)
                xv = torch.cat((xv, g_prompt[:, idx * 2 + 1, :, :].clone()), dim = 1)
            if n in self.pos_e_prompt:
                idx = self.pos_e_prompt.index(n)
                xk = torch.cat((xk, e_prompt[:, idx * 2 + 0, :, :].clone()), dim = 1)
                xv = torch.cat((xv, e_prompt[:, idx * 2 + 1, :, :].clone()), dim = 1)

            attn = block.attn
            wght = attn.qkv.weight
            bias = attn.qkv.bias

            B, N, C = x.shape
            x  = F.linear(x,  wght[:C   ,:], bias[:C   ]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xk.shape
            xk = F.linear(xk, wght[C:2*C,:], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xv.shape
            xv = F.linear(xv, wght[2*C: ,:], bias[2*C: ]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)

            attention = (x @ xk.transpose(-2, -1)) * attn.scale
            attention = attention.softmax(dim=-1)
            attention = attn.attn_drop(attention)
            x = (attention @ xv).transpose(1, 2).reshape(B, N, C)
            x = attn.proj(x)
            x = attn.proj_drop(x)

            x = r + block.drop_path1(block.ls1(x))
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))

        return x
        
    def loss_fn(self, output, target, **kwargs):
        return F.cross_entropy(output, target) - 1 * self.simmilairty