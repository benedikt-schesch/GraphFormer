from dgl.sampling import neighbor
import torch
import dgl
from torch import nn
import random
import time
from torch.nn.modules.activation import ReLU
import torch_geometric.data
import torch
from torch_geometric.loader import cluster
from torch_scatter import scatter_max, scatter_add
from torch import nn
from src.models.modules import Attention, FeedForward, PreNorm
from torch_geometric.nn import Sequential, GATConv


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        mlp_ratio=4.0,
        attn_dropout=0.0,
        dropout=0.0,
        qkv_bias=True,
        revised=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        assert isinstance(
            mlp_ratio, float
        ), "MLP ratio should be an integer for valid "
        mlp_dim = int(mlp_ratio * dim)

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                num_heads=heads,
                                qkv_bias=qkv_bias,
                                attn_drop=attn_dropout,
                                proj_drop=dropout,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout_rate=dropout,),
                        )
                        if not revised
                        else FeedForward(
                            dim, mlp_dim, dropout_rate=dropout, revised=True,
                        ),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class GraphTransformer(nn.Module):
    def __init__(self,embedding_dim,num_features,num_clases,device):
        super().__init__()
        self.to_class = nn.LazyLinear(num_clases)
        self.embedding_dim = embedding_dim
        self.device = device
        self.target_cluster_size = 16
        self.embeddings_layer = nn.LazyLinear(embedding_dim)
        self.transformer = Transformer(depth=8,heads=16,dim=embedding_dim)
        self.class_token = torch.nn.Embedding(1, embedding_dim)
        self.input_layer = nn.LazyLinear(embedding_dim)
        self.conv1 = GATConv(embedding_dim,embedding_dim)
        self.conv2 = GATConv(embedding_dim,embedding_dim)



    def forward(self, g):
        g.x = self.input_layer(g.x)
        g.x = self.conv1(g.x,g.edge_index)
        g.x = self.conv2(g.x,g.edge_index)
        clusters = torch_geometric.data.ClusterData(g,num_parts=int(g.num_nodes/self.target_cluster_size),log=False)
        cluster_embeddings = [self.class_token(torch.tensor([0]))[0]]+[torch.mean(self.embeddings_layer(cluster.x),0) for cluster in clusters]
        cluster_embeddings = torch.stack(cluster_embeddings)
        result = self.transformer(cluster_embeddings.view((1,cluster_embeddings.shape[0],cluster_embeddings.shape[1])))
        return self.to_class(result[0][0].view(1,-1))