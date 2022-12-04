import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from layers.Prompt import Prompt
from models.L2P import L2P


class DualPrompt(nn.Module):
    def __init__(self,
                 pos_g_prompt   : tuple = (),
                 len_g_prompt   : int   = 5,
                 pos_e_prompt   : tuple = (),
                 len_e_prompt   : int   = 20,
                 prompt_func    : str   = None,
                 task_num       : int   = 10,
                 class_num      : int   = 100,
                 lambd          : float = 1.0,
                 backbone_name  : str   = None,
                 **kwargs):
        super().__init__()

        if backbone_name is None:
            raise ValueError('backbone_name must be specified')

        self.register_buffer('pos_g_prompt', torch.tensor(pos_g_prompt, dtype= torch.int64))
        self.register_buffer('pos_e_prompt', torch.tensor(pos_e_prompt, dtype= torch.int64))
        self.register_buffer('similarity', torch.zeros(1))
        self.register_buffer('mask', torch.zeros(class_num))
        
        self.lambd      = lambd
        self.class_num  = class_num

        self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=class_num))
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.head.weight.requires_grad = True
        self.backbone.head.bias.requires_grad   = True

        self.tasks = []
       
        self.len_g_prompt = len_g_prompt
        self.len_e_prompt = len_e_prompt
        g_pool = 1
        e_pool = task_num
        self.g_length = len(pos_g_prompt) if pos_g_prompt else 0
        self.e_length = len(pos_e_prompt) if pos_e_prompt else 0
        
        if prompt_func == 'prompt_tuning':
            self.prompt_func = self.prompt_tuning
            self.g_prompt = None if len(pos_g_prompt) == 0 else Prompt(g_pool, 1, self.g_length * self.len_g_prompt, self.backbone.num_features, batchwise_selection = False)
            self.e_prompt = None if len(pos_e_prompt) == 0 else Prompt(e_pool, 1, self.e_length * self.len_e_prompt, self.backbone.num_features, batchwise_selection = False)

        elif prompt_func == 'prefix_tuning':
            self.prompt_func = self.prefix_tuning
            self.g_prompt = None if len(pos_g_prompt) == 0 else Prompt(g_pool, 1, 2 * self.g_length * self.len_g_prompt, self.backbone.num_features, batchwise_selection = False)
            self.e_prompt = None if len(pos_e_prompt) == 0 else Prompt(e_pool, 1, 2 * self.e_length * self.len_e_prompt, self.backbone.num_features, batchwise_selection = False)

        else: raise ValueError('Unknown prompt_func: {}'.format(prompt_func))
        
        self.task_num = task_num
        self.task_id = -1 # if _convert_train_task is not called, task will undefined

    def prompt_tuning(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      e_prompt : torch.Tensor,
                      **kwargs):

        B, _, D = x.size()
        g_prompt = g_prompt.view(B, self.g_length, self.len_g_prompt, D).clone()
        e_prompt = e_prompt.view(B, self.e_length, self.len_e_prompt, D).clone()
        g_prompt = g_prompt + self.backbone.pos_embed[:,0,:].unsqueeze(1).expand(g_prompt.size()).clone()
        e_prompt = e_prompt + self.backbone.pos_embed[:,0,:].unsqueeze(1).expand(e_prompt.size()).clone()

        for n, block in enumerate(self.backbone.blocks):
            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                x = torch.cat((x, g_prompt[:, pos_g].clone().unsqueeze(0).expand(B,-1,-1)), dim = 1)

            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                x = torch.cat((x, e_prompt[:, pos_e].clone().unsqueeze(0).expand(B,-1,-1)), dim = 1)
            x = block(x)
        return x
    
    def prefix_tuning(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      e_prompt : torch.Tensor,
                      **kwargs):

        B, N, C = x.size()
        g_prompt = g_prompt.view(B, 2 * self.g_length, self.len_g_prompt, C)
        e_prompt = e_prompt.view(B, 2 * self.e_length, self.len_e_prompt, C)
        g_prompt = g_prompt + self.backbone.pos_embed[:,0,:].unsqueeze(1).expand(g_prompt.size())
        e_prompt = e_prompt + self.backbone.pos_embed[:,0,:].unsqueeze(1).expand(e_prompt.size())

        for n, block in enumerate(self.backbone.blocks):

            xq = block.norm1(x)
            xk = xq.clone()
            xv = xq.clone()

            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                xk = torch.cat((xk, g_prompt[:, pos_g * 2 + 0]), dim = 1)
                xv = torch.cat((xv, g_prompt[:, pos_g * 2 + 1]), dim = 1)

            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                xk = torch.cat((xk, e_prompt[:, pos_e * 2 + 0]), dim = 1)
                xv = torch.cat((xv, e_prompt[:, pos_e * 2 + 1]), dim = 1)
            
            attn   = block.attn
            weight = attn.qkv.weight
            bias   = attn.qkv.bias
            
            B, N, C = xq.shape
            xq = F.linear(xq, weight[:C   ,:].clone(), bias[:C   ].clone()).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xk.shape
            xk = F.linear(xk, weight[C:2*C,:].clone(), bias[C:2*C].clone()).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xv.shape
            xv = F.linear(xv, weight[2*C: ,:].clone(), bias[2*C: ].clone()).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)

            attention = (xq @ xk.transpose(-2, -1)) * attn.scale
            attention = attention.softmax(dim=-1)
            attention = attn.attn_drop(attention)

            attention = (attention @ xv).transpose(1, 2).reshape(B, N, C)
            attention = attn.proj(attention)
            attention = attn.proj_drop(attention)

            x = x + block.drop_path1(block.ls1(attention))
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))

        return x

    def forward(self, inputs : torch.Tensor) :

        with torch.no_grad():
            x = self.backbone.patch_embed(inputs)
            B, N, D = x.size()

            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            token_appended = torch.cat((cls_token, x), dim=1)
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
            query = self.backbone.blocks(x)
            query = self.backbone.norm(query)[:, 0].clone()


        if self.g_prompt is not None:
            # g_p = self.g_prompt.prompts[0].expand(B, -1, -1).clone()
            g_s, g_p = self.g_prompt(query)
        else:
            g_p = None
        if self.e_prompt is not None:    
            if self.training:
                e_s = F.cosine_similarity(query.unsqueeze(1), self.e_prompt.key, dim = -1)[:, self.task_id].clone()
                e_p = self.e_prompt.prompts[self.task_id].expand(B, -1, -1).clone()
                self.e_prompt.counter[self.task_id] += 1
            else:
                e_s, e_p = self.e_prompt(query)
                # e_s = F.cosine_similarity(query.unsqueeze(1), self.e_prompt.key, dim = -1)
                # idx = torch.argmax(e_s, dim = -1)
                # e_s = e_s[idx].clone()
                # e_p = self.e_prompt.prompts[idx].clone().expand(B, -1, -1)
                # self.e_prompt.counter[idx] += 1
        else:
            e_p = None
            e_s = 0

        x = self.prompt_func(token_appended + self.backbone.pos_embed, g_p, e_p)
        x = self.backbone.norm(x)
        x = self.backbone.head(x[:, 0])

        self.similairty = e_s.sum()
        return x

    def convert_train_task(self, task : torch.Tensor, **kwargs):
        flag = -1
        for n, t in enumerate(self.tasks):
            if torch.equal(t, task):
                flag = n
                break
        if flag == -1:
            self.tasks.append(task)
            self.task_id = len(self.tasks) - 1
            with torch.no_grad():
                self.e_prompt.prompts[self.task_id] = self.e_prompt.prompts[self.task_id - 1].clone()
        else :
            self.task_id = flag
        # self.task_id = 0
        self.mask += -torch.inf
        self.mask[task] = 0
        return
        
    def get_count(self):
        return self.e_prompt.update()

    def loss_fn(self, output, target):
        B, C = output.size()
        return F.cross_entropy(output, target) + self.lambd * self.similarity