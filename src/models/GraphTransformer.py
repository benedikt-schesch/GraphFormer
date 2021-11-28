from dgl.sampling import neighbor
import torch
import dgl
from torch import nn
import random
import time
from torch.nn.modules.activation import ReLU

import torch
from torch_scatter import scatter_max, scatter_add

class GraphTransformer(nn.Module):
    def __init__(self,embedding_dim,num_features,num_clases,device):
        super().__init__()
        self.linear = nn.Linear(2,num_clases)
        self.embedding_dim = embedding_dim
        self.device = device


    def forward(self, x ):
        return self.linear(torch.tensor([0.,1.]))
