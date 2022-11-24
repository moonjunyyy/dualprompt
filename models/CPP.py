from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Prompt import Prompt
from models.L2P import L2P

class CPP(L2P):
    def __init__(self,
                 len_prompt     : int   = 8,
                 num_centroids  : int   = 5,
                 num_neighbors  : int   = 20,
                 prompt_func    : str   = None,
                 task_num       : int   = 10,
                 class_num      : int   = 100,
                 lambd          : float = 1.0,
                 backbone_name  : str   = None,
                 **kwargs):

        super().__init__(class_num = class_num,
                         lambd     = lambd,
                         backbone_name = backbone_name,**kwargs)
        del(self.prompt)
        del(self.selection_size)
        del(self.prompt_len)

        self.tasks = []

        self.register_buffer('pos_g_prompt', torch.tensor(pos_g_prompt, dtype= torch.int64))
        self.register_buffer('pos_e_prompt', torch.tensor(pos_e_prompt, dtype= torch.int64))
        
        self.len_prompt = len_prompt
        
        e_pool = class_num
        self.g_length = len(self.backbone.blocks)
        
        elif prompt_func == 'prefix_tuning':
            self.prompt_func = self.prefix_tuning
            self.prompt = Prompt(e_pool, 1, 2 * self.e_length * self.len_e_prompt, self.backbone.num_features, batchwise_selection = True)

        else: raise ValueError('Unknown prompt_func: {}'.format(prompt_func))
        
        self.task_num = task_num
        self.task_id = -1 # if _convert_train_task is not called, task will undefined
    
    def prefix_tuning(self, x : torch.Tensor, g_prompt : torch.Tensor, e_prompt : torch.Tensor, **kwargs):

        B, N, C = x.size()
        g_prompt = g_prompt.view(B, self.g_length, self.len_g_prompt, C)
        e_prompt = e_prompt.view(B, self.e_length, self.len_e_prompt, C)
        g_prompt = g_prompt + self.pos_embed[:,0,:].unsqueeze(1).expand(g_prompt.size())
        e_prompt = e_prompt + self.pos_embed[:,0,:].unsqueeze(1).expand(e_prompt.size())

        for n, block in enumerate(self.backbone.blocks):

            xq = block.norm1(x)
            xk = xq.clone()
            xv = xq.clone()

            pos_g = ((self.pos_g_prompt == n).nonzero()).squeeze()
            if pos_g.numel() != 0:
                xk = torch.cat((xk, g_prompt[:, pos_g * 2 + 0].clone().unsqueeze(0).expand(B,-1,-1)), dim = 1)
                xv = torch.cat((xv, g_prompt[:, pos_g * 2 + 1].clone().unsqueeze(0).expand(B,-1,-1)), dim = 1)

            pos_e = ((self.pos_e_prompt == n).nonzero()).squeeze()
            if pos_e.numel() != 0:
                xk = torch.cat((xk, e_prompt[:, pos_e * 2 + 0].clone().unsqueeze(0).expand(B,-1,-1)), dim = 1)
                xv = torch.cat((xv, e_prompt[:, pos_e * 2 + 1].clone().unsqueeze(0).expand(B,-1,-1)), dim = 1)
            
            attn   = block.attn
            weight = attn.qkv.weight
            bias   = attn.qkv.bias
            
            B, N, C = xq.shape
            xq = F.linear(xq, weight[:C   ,:], bias[:C   ]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xk.shape
            xk = F.linear(xk, weight[C:2*C,:], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xv.shape
            xv = F.linear(xv, weight[2*C: ,:], bias[2*C: ]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)

            attention = (xq @ xk.transpose(-2, -1)) * attn.scale
            attention = attention.softmax(dim=-1)
            attention = attn.attn_drop(attention)

            attention = (attention @ xv).transpose(1, 2).reshape(B, N, C)
            attention = attn.proj(attention)
            attention = attn.proj_drop(attention)

            x = x + block.drop_path(attention)
            x = x + block.drop_path(block.mlp(block.norm2(x)))
        return x

    def forward(self, inputs : torch.Tensor) :
        x = self.backbone.patch_embed(inputs)
        B, N, D = x.size()

        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        t = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(t + self.backbone.pos_embed)
        q = self.backbone.blocks(x)
        q = self.backbone.norm(q)[:, 0, :].clone()

        g_s, g_p = self.g_prompt(q)
        if self.training:
            e_s = F.cosine_similarity(q, self.e_prompt.key[self.task_id], dim = -1)
            e_p = self.e_prompt.prompts[self.task_id].clone().expand(B, -1, -1)
        else:
            e_s, e_p = self.e_prompt(q)

        x = self.prompt_func(t + self.pos_embed, g_p, e_p)
        x = self.backbone.norm(x)
        x = self.backbone.head(x[:, 0].clone())
        self.simmilairty = e_s.sum()
        return x

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
        
        return self.e_prompt.update()
