import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class ViTAutoEncoder(nn.Module):
    def __init__(self,
                 class_num      : int   = 100,
                 backbone_name  : str   = None,
                 **kwargs) -> None:
        super().__init__()
        if backbone_name is None:
            raise ValueError('backbone_name must be specified')
        self.add_module('encoder', timm.create_model(backbone_name, pretrained=True, num_classes=class_num))
        for param in self.encoder.parameters():
            param.requires_grad = True

        self.msk_token = nn.Parameter(torch.randn(1, 1, self.encoder.num_features), requires_grad=True)
        self.decoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=768, nhead=12), num_layers=12)
        
    def forward(self, x):

        image = self.encoder.patch_embed(x)
        B, N, D = image.size()

        cls_token = self.encoder.cls_token.expand(B, -1, -1)
        token_appended = torch.cat((cls_token, image), dim=1)
        x = self.encoder.pos_drop(token_appended + self.encoder.pos_embed)
        
        encoded = self.encoder.blocks(x)
        encoded = self.encoder.norm(encoded)[:, 0].clone()

        msk_token = self.msk_token.expand(B, N, -1)
        x = torch.cat((encoded.unsqueeze(1), msk_token), dim=1)
        x = self.decoder(x)
        
        self.image = image
        self.reconstructed = x[:, 1:]

        return self.encoder.head(encoded)

    def loss_fn(self, x, y):
        return F.cross_entropy(x, y) + F.mse_loss(self.image, self.reconstructed)

    