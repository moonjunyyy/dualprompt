from typing import TypeVar

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.kmeans import KMeans

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
        self.mean     = nn.Parameter(F.normalize(torch.randn(self.pool_size, self.backbone.num_features)), requires_grad=False)
        # self.register_buffer('mean',F.normalize(torch.randn(self.pool_size, self.backbone.num_features)))
        self.variance = nn.Parameter((torch.ones((self.backbone.num_features,self.backbone.num_features))
                                         + torch.eye(self.backbone.num_features)
                                        ).repeat(self.pool_size,1,1), requires_grad = False)
        self.query_buffer = None

        self.register_buffer('frequency', torch.ones(self.pool_size))
        self.register_buffer('counter',   torch.zeros(self.pool_size))
        self.register_buffer('sample',    torch.zeros((pool_size, class_num)))
        self.register_buffer('mask',      torch.zeros(class_num))

    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:

        torch.cuda.empty_cache()
        self.distribution = torch.distributions.MultivariateNormal(self.mean, self.variance.abs() + torch.eye(self.backbone.num_features).to(self.mean.device))
        img = self.backbone.patch_embed(inputs)

        B, N, D = img.size()
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        t = torch.cat((cls_token, img), dim=1)
        x = self.backbone.pos_drop(t + self.backbone.pos_embed)
        
        with torch.no_grad():
            q = self.backbone.blocks(x)
            q = self.backbone.norm(q)[:, 0].clone()
            # if self.training:
            #     psudo_samples = self.distribution.sample((self.psudo_sapmle,))
            #     psudo_samples = torch.cat((q.unsqueeze(1).expand(-1, self.pool_size, -1), psudo_samples), dim=0)
            #     prob = self.distribution.log_prob(psudo_samples)
            if self.training:
                if self.query_buffer is None:
                    self.query_buffer = q.detach().clone()
                else:
                    self.query_buffer = torch.cat((q.detach().clone(), self.query_buffer))
                # prob = self.distribution.log_prob(q.unsqueeze(1))
                # B, P
                # topk = prob[:B] * (self.frequency / self.frequency.max() * 16).exp() #Diversed Selection
            topk = self.distribution.log_prob(q.unsqueeze(1))
            # if self.training:
            _ ,topk = topk.topk(self.selection_size, dim=-1, largest=True, sorted=True)
            self.counter += torch.bincount(topk.contiguous().view(-1), minlength = self.pool_size)
            prompts = self.prompt.repeat(B, 1, 1, 1).gather(1, topk.unsqueeze(-1).unsqueeze(-1).expand(B, -1, self.prompt_len, self.backbone.num_features))
        # B, P
        prompts = prompts.contiguous().view(B, self.selection_size * self.prompt_len, D)
        prompts = prompts + self.backbone.pos_embed[:,0].expand(self.selection_size * self.prompt_len, -1)
        # x = self.backbone.pos_drop(t + self.backbone.pos_embed)
        # sel = torch.rand(B, N).argsort(dim=1).to(x.device)
        pos_embed = self.backbone.pos_embed.clone()
        x = self.backbone.pos_drop(t + pos_embed)

        # img = img + self.backbone.pos_embed[:,1:].clone()
        # if self.training:
        #     sel = torch.rand((B, N),device=x.device).argsort(dim=1)
        #     img = torch.concat((
        #         img.gather(1, sel[:, :int(N*0.5)].unsqueeze(-1).expand(-1, -1, D)),
        #         img.gather(1, sel[:, int(N*0.5):].unsqueeze(-1).expand(-1, -1, D)))
        #         , dim = 0)
        #     x = torch.cat((x[:,:1].repeat(2,1,1), prompts.repeat(2,1,1), img), dim=1)
        # else:
        #     x = torch.cat((x[:,:1], prompts, img), dim=1)
        x = torch.cat((x[:,:1], prompts, img), dim=1)

        x = self.backbone.pos_drop(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        x = x[:, 1:self.selection_size*self.prompt_len+1].clone()

        if self.training:
            with torch.no_grad():
                kmeans = KMeans(n_clusters=self.pool_size).fit(self.query_buffer)
                _labels = kmeans.labels_
                _cluster_centers = kmeans.cluster_centers_
                
                dist = torch.cdist(self.mean, _cluster_centers)
                idx = dist.reshape(-1).argsort().reshape(self.pool_size, -1)
                flag_pmtpool = torch.ones(self.pool_size, dtype=torch.bool)
                flag_cluster = torch.ones(self.pool_size, dtype=torch.bool)
                # P, P
                for i in range(self.pool_size * self.pool_size):
                    pos = (idx == i).nonzero(as_tuple=True)
                    if flag_pmtpool[pos[0]] and flag_cluster[pos[1]]:
                        flag_pmtpool[pos[0]] = False
                        flag_cluster[pos[1]] = False
                        self.mean[pos[0]] = _cluster_centers[pos[1]]
                # self.mean = _cluster_centers.detach().clone()
        if self.training:
            # self.distribution = torch.distributions.MultivariateNormal(self.mean, self.variance.abs())
            # prob = self.distribution.log_prob(self.query_buffer.unsqueeze(1))

            self.prob_loss = 0
            dist = torch.cdist(self.mean, _cluster_centers)
            idx = dist.reshape(-1).argsort().reshape(self.pool_size, -1)
            flag_pmtpool = torch.ones(self.pool_size, dtype=torch.bool)
            flag_cluster = torch.ones(self.pool_size, dtype=torch.bool)
            for i in range(self.pool_size*self.pool_size):
                pos = (idx == i).nonzero(as_tuple=True)
                if flag_pmtpool[pos[0]] and flag_cluster[pos[1]]:
                    flag_pmtpool[pos[0]] = False
                    flag_cluster[pos[1]] = False
                    self.variance[pos[0]] = torch.cov(self.query_buffer[_labels == pos[1]].t(), correction=0)
            
            self.prob_loss     = self.prob_loss / self.query_buffer.size(0)
            self.var_l2_loss   =  (self.variance[topk[:,:3]]).pow(2).mean()
            # self.var_l2_loss   =  (self.variance[topk[:,:2]] - torch.tril(torch.ones((self.backbone.num_features,self.backbone.num_features), device=self.variance.device))).pow(2).mean()
            self.distance_loss =  (self.mean[topk[:,:3]].unsqueeze(2).expand(-1,-1,3,-1) - self.mean[topk[:,:3]].unsqueeze(1)).norm(dim=-1).mean()
        else:
            self.prob_loss     = 0
            self.var_l2_loss   = 0
            self.distance_loss = 0

        if self.training:
            with torch.no_grad():
                _buffer = torch.zeros((1, self.backbone.num_features), device=self.query_buffer.device)
                for i in range(self.pool_size):
                        _labels_i = _labels.eq(i)
                        if _labels_i.sum() > self.psudo_sapmle:
                            # get random samples from query buffer
                            _buffer = torch.cat((
                                _buffer,
                                self.query_buffer[_labels.eq(i)].index_select(0, torch.randperm(_labels.eq(i).sum(), device=self.query_buffer.device)[:self.psudo_sapmle]))
                                , dim=0)
                            # get the most closest to cluster center from query buffer
                            # _, idx = (self.query_buffer[_labels_i] - _cluster_centers[i]).norm(dim=1).topk(self.psudo_sapmle, largest=False, sorted=True)
                            # _buffer = torch.cat((_buffer, self.query_buffer[_labels_i][idx]), dim=0)
                        else:
                            _buffer = torch.cat((_buffer, self.query_buffer[_labels.eq(i)]), dim=0)
                self.query_buffer = _buffer[1:].detach().clone()
        # if self.training:
        #     self.imgloss = x[:B].clone().transpose(1,2) @ x[B:].clone()
        #     self.imgloss = self.imgloss / self.imgloss.max()
        #     self.imgloss = -(self.imgloss.diagonal(dim1=1, dim2=2).exp().sum() / self.imgloss.exp().sum()).log()        
        #     x = x.mean(dim=1)
        # else:
        #     x = x.mean(dim=1)
        x = x.mean(dim=1)
        x = self.backbone.head(x)

        if self.training:
            # x = (x[:B].clone() + x[B:].clone())/2 
            x = x + self.mask
        
        return x
    
    def loss_fn(self, output, target):
        B, C = output.size()
        return F.cross_entropy(output, target) # + self.lambd * self.prob_loss # - self.zetta * self.distance_loss # + self.xi * self.var_l2_loss  # + self.zetta * self.imgloss

    def convert_train_task(self, task : torch.Tensor, **kwargs):
        self.mask += -torch.inf
        self.mask[task.to(self.mask.device)] = 0
        return self.frequency - 1

    def get_count(self):
        if self.training:
            self.frequency += self.counter
            self.counter.zero_()
            return self.frequency - 1
        else:
            count = self.counter.clone()
            self.counter.zero_()
            return count

    def train(self: T, mode: bool = True, **kwargs) -> T:
        ten = super().train(mode)
        self.backbone.eval()
        return ten
