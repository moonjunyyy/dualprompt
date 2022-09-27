from models.L2P import L2P
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledL2P(L2P):
    def __init__(self, tau = 2, _scale_simmilarity = False, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.tau = tau
        self._scale_simmilarity = _scale_simmilarity
    
    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:

        #self.prompt.prompts.data = F.normalize(self.prompt.prompts.data, dim = -1)
        x = self.backbone.patch_embed(inputs)

        B, N, D = x.size()
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        t = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(t + self.backbone.pos_embed)
        q = self.backbone.blocks(x)
        q = self.backbone.norm(q)[:, 0].clone()

        # Scaling the Prompts
        s, p = self.prompt(q)
        scale = ((s - 1) * self.tau).exp()
        if self._scale_simmilarity:
            s = s * scale
        p = p * scale.unsqueeze(-1).unsqueeze(-1)
        self.simmilarity = s.sum()

        p = p.contiguous().view(B, self.selection_size * self.prompt_len, D)
        p = p + self.pos_embed[:,0].clone().expand(self.selection_size * self.prompt_len, -1)

        x = self.backbone.pos_drop(t + self.pos_embed)
        if self._cls_at_front:
            x = torch.cat((x[:,0].unsqueeze(1), p, x[:,1:]), dim=1)
        else :
            x = torch.cat((p, x), dim=1)

        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)

        if self._cls_at_front:
            x = x[:, 1:self.selection_size * self.prompt_len + 1].clone()
        else :
            x = x[:, :self.selection_size * self.prompt_len].clone()
        x = x.mean(dim=1)
        x = self.backbone.head(x)

        if self.training:
            x = x + self.mask
        return x
    