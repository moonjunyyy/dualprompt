from turtle import forward
import torch
import torch.nn as nn
from utils.kmeans import KMeans
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.L2P import L2P

class CPP(L2P):
    def __init__(self,
                 len_prompt     : int   = 8,
                 num_centroids  : int   = 5,
                 num_neighbors  : int   = 20,
                 num_tasks      : int   = 10,
                 class_num      : int   = 100,
                 backbone_name  : str   = None,
                 **kwargs):

        super().__init__(class_num = class_num,
                         lambd     = 0.5,
                         backbone_name = backbone_name,**kwargs)
        del(self.prompt)
        del(self.selection_size)
        del(self.prompt_len)

        self.backbone.head.weight.requires_grad = False
        self.backbone.head.bias.requires_grad = False

        self.tasks = []
        self.num_centroids = num_centroids
        self.num_neighbors = num_neighbors
        self.len_prompt    = len_prompt
        
        self.prompt_number = len(self.backbone.blocks)
                
        self.key_prototypes = nn.Parameter(torch.randn(self.class_num, self.num_centroids, self.backbone.num_features), requires_grad=False)
        self.prompt         = nn.Parameter(torch.randn(self.class_num, self.prompt_number, 2, self.len_prompt, self.backbone.num_features))
        self.val_prototypes = nn.Parameter(torch.randn(self.class_num, self.num_centroids, self.backbone.num_features), requires_grad=False)

        self.mlp_layers = nn.Sequential(
            nn.Linear(self.backbone.num_features, 2048),
            nn.ReLU(inplace = True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace = True),
            nn.Linear(2048, self.backbone.num_features),
        )

        self.task_num = num_tasks
        self.task_id = -1 # if _convert_train_task is not called, task will undefined

    def pre_forward(self, dataloader : DataLoader, **kwargs) -> torch.Tensor:
        
        self.mlp_layers[0].reset_parameters() 
        self.mlp_layers[2].reset_parameters() 
        self.mlp_layers[4].reset_parameters() 

        embedding_buffer = torch.empty((0, self.backbone.num_features), device=self.prompt.device)
        class_buffer     = torch.empty((0,), device = self.prompt.device, dtype=torch.long)

        for i, (x, y) in enumerate(dataloader):
            x = x.to(self.prompt.device)
            y = y.to(self.prompt.device)
            
            x = self.backbone.patch_embed(x)
            B, N, C = x.size()
            x = self.embedding(x)

            embedding_buffer = torch.cat((embedding_buffer, x), dim = 0)
            class_buffer     = torch.cat((class_buffer, y), dim = 0)

        unique = class_buffer.unique()
        for c in unique:
            embeddings = embedding_buffer[class_buffer == c]
            simmilarity_matrix = torch.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim = -1)
            eig = torch.linalg.eigh(torch.diag(simmilarity_matrix.sum(dim=0)) - simmilarity_matrix, UPLO='U')[1][:,1:self.num_centroids]
            kmeans = KMeans(n_clusters=self.num_centroids).fit(eig)
            for i in range(self.num_centroids):
                self.key_prototypes[c][i] = embeddings[kmeans.labels_==i].mean(dim=0)

    def post_forward(self, dataloader : DataLoader, **kwargs) -> torch.Tensor:

        embedding_buffer = torch.empty((0, self.backbone.num_features), device=self.prompt.device)
        class_buffer     = torch.empty((0,), device = self.prompt.device, dtype=torch.long)

        for i, (x, y) in enumerate(dataloader):
            x = x.to(self.prompt.device)
            y = y.to(self.prompt.device)
            prompt = self.prompt[y].clone()
            x = self.backbone.patch_embed(x)
            B, N, C = x.size()

            x = torch.cat((self.embedding(x[:int(B/2)], prompt[:int(B/2)]),self.embedding(x[int(B/2):], prompt[int(B/2):])), dim = 1)
            
            embedding_buffer = torch.cat((embedding_buffer, x), dim = 0)
            class_buffer     = torch.cat((class_buffer, y), dim = 0)

        unique = class_buffer.unique()
        for c in unique:
            embeddings = embedding_buffer[class_buffer == c]
            simmilarity_matrix = torch.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim = -1)
            eig = torch.linalg.eigh(torch.diag(simmilarity_matrix.sum(dim=0)) - simmilarity_matrix, UPLO='U')[1][:,1:self.num_centroids]
            kmeans = KMeans(n_clusters=self.num_centroids).fit(eig)
            for i in range(self.num_centroids):
                self.val_prototypes[c][i] = embeddings[kmeans.labels_==i].mean(dim=0)

    def forward(self, inputs : torch.Tensor) :

        x = self.backbone.patch_embed(inputs)
        B, N, C = x.size()

        if self.training:
            self.data_buffer = x.clone()
            return torch.rand((B,N), device = inputs.device)

        else:
            emb = self.embedding(x)
            simmilarity = torch.cosine_similarity(emb.unsqueeze(1), self.key_prototypes.reshape(-1, C).unsqueeze(0), dim = -1)
            topk, idx = simmilarity.topk(self.num_neighbors, dim = -1, largest = False, sorted = True)
            idx = idx // self.num_centroids
            prompt = self.prompt[idx]

            prompt = prompt.view(B, self.num_neighbors, self.prompt_number, 2, self.len_prompt, C)
            prompt = prompt.reshape(-1, self.prompt_number, 2, self.len_prompt, C)
            x = x.repeat(self.num_neighbors, 1, 1)
            
            y = torch.empty((0, self.backbone.num_features), device = inputs.device)
            for i in range(self.num_neighbors):
                y = torch.cat((y, self.embedding(x[i*B:(i+1)*B], prompt[i*B:(i+1)*B])), dim = 0)

            # Class prediction
            # B * C, D
            # C * CC, D
            simmilarity = F.cosine_similarity(y.unsqueeze(1), self.val_prototypes.reshape(-1, C).unsqueeze(0), dim = -1).reshape(B, self.num_neighbors, self.class_num, self.num_centroids)
            argmin = simmilarity.min(dim = -1).values.min(dim = 1).values.argmin(dim = 1)
            print(argmin)
            answer = torch.zeros((B,N), device = inputs.device)
            answer[torch.arange(B), argmin] = 1
            
            return answer

    def embedding(self, x : torch.Tensor, prompt : torch.Tensor = None, **kwargs) -> torch.Tensor:
        B, N, C = x.size()
        if prompt is None:
            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            t = torch.cat((cls_token, x), dim=1)
            x = self.backbone.pos_drop(t + self.backbone.pos_embed)
            q = self.backbone.blocks(x)
            q = self.backbone.norm(q)[:, 0].clone()
            return q
        else:
            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            t = torch.cat((cls_token, x), dim=1)
            x = self.backbone.pos_drop(t + self.backbone.pos_embed)
            prompt = prompt + self.backbone.pos_embed[:,:1,:].unsqueeze(1).unsqueeze(1).expand(prompt.size(0), -1, prompt.size(2), prompt.size(3), -1)
            for n, block in enumerate(self.backbone.blocks):
                xq = block.norm1(x)
                xk = xq.clone()
                xv = xq.clone()

                xk = torch.cat((xk, prompt[:, n, 0].clone()), dim = 1)
                xv = torch.cat((xv, prompt[:, n, 1].clone()), dim = 1)

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
            x = self.backbone.norm(x)[:, 0].clone()
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
        else :
            self.task_id = flag
        self.mask += -torch.inf
        self.mask[task] = 0
        return torch.zeros_like(task)

    def get_count(self):
        return torch.zeros((self.class_num), device = self.mask.device)

    def loss_fn(self, output, target):
        B, N, D = self.data_buffer.size()
        if self.training:
            prompt = torch.empty((0, self.prompt_number, 2, self.len_prompt, self.backbone.num_features), device = output.device)
            for i, n in enumerate(target):
                prompt = torch.cat((prompt, self.prompt[n].clone().unsqueeze(0)), dim = 0)
            x = self.embedding(self.data_buffer, prompt)
            x = self.mlp_layers(x)

            loss = 0
            for cls in self.tasks[-1]:
                positive = x[target == cls].clone()
                positive_sum = F.cosine_similarity(positive.unsqueeze(1), torch.cat((positive, self.key_prototypes[cls].clone())).unsqueeze(0), dim = -1).exp().sum()
                negative = x[target != cls].clone()
                for past_cls in self.tasks[:-1]:
                    for clss in past_cls:
                        negative = torch.cat((negative, self.val_prototypes[clss].clone()), dim = 0)
                negative_sum = F.cosine_similarity(positive.unsqueeze(1), negative.unsqueeze(0), dim = -1).exp().sum()
                loss += -torch.log(positive_sum /(negative_sum + 1e-6) + 1e-6)
            return loss / self.tasks[-1].size(0)

        else:
            return torch.zeros(1, device = output.device)
