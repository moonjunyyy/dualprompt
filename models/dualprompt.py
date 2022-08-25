import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.prompt import Prompt

class DualPrompt(nn.Module):
    def __init__(self,
                 pool_size      : int,
                 selection_size : int,
                 pos_g_prompt   : tuple,
                 pos_e_prompt   : tuple,
                 dimention      : int,
                 model_fn       : function,
                 *args, **kwargs):
        super(DualPrompt, self).__init__()
        
        self.backbone = model_fn()
        self.g_prompt = Prompt(pool_size, selection_size, len(pos_g_prompt), dimention)
        self.e_prompt = Prompt(pool_size, selection_size, len(pos_e_prompt), dimention)
        self.pos_g_prompt = pos_g_prompt
        self.pos_e_prompt = pos_e_prompt

    def forward(self, query : torch.Tensor, *args, **kwargs):
        query = query.view(-1, 1, self.dimention)
        
        return