from dgl.sampling import neighbor
import torch
import dgl
from torch import nn
import random
import time
from random import sample
import numpy as np
from numba import jit
from torch.nn.modules.activation import ReLU
import torch_geometric.data
import torch
from torch_geometric.loader import cluster
from torch_scatter import scatter_max, scatter_add
from torch import nn
from src.models.modules import Attention, FeedForward, PreNorm
from torch_geometric.nn import Sequential, GATConv
from itertools import groupby


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
        self.target_cluster_size = 32
        self.embeddings_layer = nn.LazyLinear(embedding_dim)
        self.transformer = Transformer(depth=8,heads=8,dim=embedding_dim)
        self.class_token = nn.Embedding(1, embedding_dim)
        self.input_layer = nn.LazyLinear(embedding_dim)
        self.conv1 = GATConv(embedding_dim,embedding_dim)
        self.conv2 = GATConv(embedding_dim,embedding_dim)
        self.node_embedding_layer = nn.Embedding(1,embedding_dim)
    
    def generate_clusters(self,edge_index, num_nodes, adj):
        #edges = edge_index.transpose(1,0).tolist()
        #adj = {k: [v[1] for v in g] for k, g in groupby(sorted(edges), lambda e: e[0])}
        clusters = np.full(num_nodes,0)
        unvisited_nodes = set({i for i in range(num_nodes)})
        cluster_id = 0
        while len(unvisited_nodes) != 0:
            neighbs = set({})
            cluster = [random.sample(unvisited_nodes,1)[0]]
            unvisited_nodes.remove(cluster[-1])
            neighbs.update(adj[cluster[-1]])
            neighbs = neighbs.intersection(unvisited_nodes)
            clusters[cluster[-1]] = cluster_id
            cluster_is_full = False
            while not cluster_is_full:
                if len(neighbs) == 0: #We have no neighbors so we add some random unvisited node
                    if len(unvisited_nodes) == 0:
                        cluster_is_full = True
                        break
                    cluster.append(random.sample(unvisited_nodes,1)[0])
                    unvisited_nodes.remove(cluster[-1])
                    neighbs.update(adj[cluster[-1]])
                    neighbs = neighbs.intersection(unvisited_nodes)
                    clusters[cluster[-1]] = cluster_id
                    if len(cluster) == self.target_cluster_size:
                        cluster_is_full = True
                        break
                while len(neighbs) > 0: #We do have neighbors so we add one randomly
                    cluster.append(random.sample(neighbs,1)[0])
                    neighbs.remove(cluster[-1])
                    unvisited_nodes.remove(cluster[-1])
                    neighbs.update(adj[cluster[-1]])
                    neighbs = neighbs.intersection(unvisited_nodes)
                    clusters[cluster[-1]] = cluster_id
                    if len(cluster) == self.target_cluster_size:
                        cluster_is_full = True
                        break
            cluster_id+=1
        return clusters, cluster_id

    def forward(self, g ,adj):
        if g.x == None: #In case we don't have any Node features we learn an embedding
            x = self.node_embedding_layer(torch.tensor([0]))[0].repeat(g.num_nodes,1)
        else:
            x = g.x
        clusters, num_clusters = self.generate_clusters(g.edge_index,g.num_nodes,adj)
        x = self.input_layer(x)
        x = self.conv1(x,g.edge_index)
        #x = self.conv2(x,g.edge_index)
        cluster_embeddings = []
        for i in range(num_clusters-1):
            cluster_elements = np.nonzero(clusters==i)[0]
            cluster_embeddings.append(self.embeddings_layer(x[cluster_elements].view(-1)))
        
        cluster_elements = np.nonzero(clusters==(num_clusters-1))[0]
        cluster_embeddings.append(self.embeddings_layer(torch.cat((x[cluster_elements].view(-1),
        torch.zeros(self.embedding_dim*self.target_cluster_size-x[cluster_elements].view(-1).shape[0])))))

        cluster_embeddings = [self.class_token(torch.zeros(1).type(torch.LongTensor)).view(-1)] + cluster_embeddings
        cluster_embeddings = torch.stack(cluster_embeddings)
        result = self.transformer(cluster_embeddings.view((1,cluster_embeddings.shape[0],cluster_embeddings.shape[1])))
        return self.to_class(result[0][0].view(1,-1))



class ConvAggregation(nn.Module):
    def __init__(self,embedding_dim,num_features,num_clases,device):
        super().__init__()
        self.to_class = nn.LazyLinear(num_clases)
        self.embedding_dim = embedding_dim
        self.device = device
        self.embeddings_layer = nn.LazyLinear(embedding_dim)
        self.input_layer = nn.LazyLinear(embedding_dim)
        self.conv1 = GATConv(embedding_dim,embedding_dim)
        self.conv2 = GATConv(embedding_dim,embedding_dim)
        self.conv3 = GATConv(embedding_dim,embedding_dim)
        self.conv4 = GATConv(embedding_dim,embedding_dim)
        self.node_embedding_layer = nn.Embedding(1,embedding_dim)

    def forward(self, g ,adj):
        if g.x == None: #In case we don't have any Node features we learn an embedding
            x = self.node_embedding_layer(torch.tensor([0]))[0].repeat(g.num_nodes,1)
        else:
            x = g.x
        x = self.input_layer(x)
        x = self.conv1(x,g.edge_index)
        x = self.conv2(x,g.edge_index)
        x = self.conv3(x,g.edge_index)
        x = self.conv4(x,g.edge_index)

        x = torch.mean(x,0)
        return self.to_class(x).view(1,-1)
    


class FullGraphTransformer(nn.Module):
    def __init__(self,embedding_dim,num_features,num_clases,device):
        super().__init__()
        self.to_class = nn.LazyLinear(num_clases)
        self.embedding_dim = embedding_dim
        self.device = device
        self.target_cluster_size = 32
        self.embeddings_layer = nn.LazyLinear(embedding_dim)
        self.transformer = Transformer(depth=8,heads=8,dim=embedding_dim)
        self.class_token = nn.Embedding(1, embedding_dim)
        self.input_layer = nn.LazyLinear(embedding_dim)
        self.conv1 = GATConv(embedding_dim,embedding_dim)
        self.conv2 = GATConv(embedding_dim,embedding_dim)
        self.node_embedding_layer = nn.Embedding(1,embedding_dim)

    def forward(self, g ,adj):
        if g.x == None: #In case we don't have any Node features we learn an embedding
            x = self.node_embedding_layer(torch.tensor([0]))[0].repeat(g.num_nodes,1)
        else:
            x = g.x
        #clusters, num_clusters = self.generate_clusters(g.edge_index,g.num_nodes,adj)
        x = self.input_layer(x)
        #x = self.conv1(x,g.edge_index)
        #x = self.conv2(x,g.edge_index)
        #cluster_embeddings = []
        #for i in range(num_clusters-1):
        #    cluster_elements = np.nonzero(clusters==i)[0]
        #    cluster_embeddings.append(self.embeddings_layer(x[cluster_elements].view(-1)))
        
        #cluster_elements = np.nonzero(clusters==(num_clusters-1))[0]
        
        x = torch.cat((self.class_token(torch.zeros(1).type(torch.LongTensor)),x))
        result = self.transformer(x.view((1,x.shape[0],x.shape[1])))
        return self.to_class(result[0][0].view(1,-1))



class RandomGraphTransformer(nn.Module):
    def __init__(self,embedding_dim,num_features,num_clases,device):
        super().__init__()
        self.to_class = nn.LazyLinear(num_clases)
        self.embedding_dim = embedding_dim
        self.device = device
        self.target_cluster_size = 32
        self.embeddings_layer = nn.LazyLinear(embedding_dim)
        self.transformer = Transformer(depth=8,heads=8,dim=embedding_dim)
        self.class_token = nn.Embedding(1, embedding_dim)
        self.input_layer = nn.LazyLinear(embedding_dim)
        self.conv1 = GATConv(embedding_dim,embedding_dim)
        self.conv2 = GATConv(embedding_dim,embedding_dim)
        self.node_embedding_layer = nn.Embedding(1,embedding_dim)
    
    def generate_clusters(self,edge_index, num_nodes, adj):
        #edges = edge_index.transpose(1,0).tolist()
        #adj = {k: [v[1] for v in g] for k, g in groupby(sorted(edges), lambda e: e[0])}
        clusters = np.full(num_nodes,0)
        unvisited_nodes = set({i for i in range(num_nodes)})
        cluster_id = 0
        while len(unvisited_nodes) != 0:
            neighbs = set({})
            cluster = [random.sample(unvisited_nodes,1)[0]]
            unvisited_nodes.remove(cluster[-1])
            neighbs.update(adj[cluster[-1]])
            neighbs = neighbs.intersection(unvisited_nodes)
            clusters[cluster[-1]] = cluster_id
            cluster_is_full = False
            while not cluster_is_full:
                if len(unvisited_nodes) == 0:
                    cluster_is_full = True
                    break
                cluster.append(random.sample(unvisited_nodes,1)[0])
                unvisited_nodes.remove(cluster[-1])
                neighbs.update(adj[cluster[-1]])
                neighbs = neighbs.intersection(unvisited_nodes)
                clusters[cluster[-1]] = cluster_id
                if len(cluster) == self.target_cluster_size:
                    cluster_is_full = True
                    break
            cluster_id+=1
        return clusters, cluster_id

    def forward(self, g ,adj):
        if g.x == None: #In case we don't have any Node features we learn an embedding
            x = self.node_embedding_layer(torch.tensor([0]))[0].repeat(g.num_nodes,1)
        else:
            x = g.x
        clusters, num_clusters = self.generate_clusters(g.edge_index,g.num_nodes,adj)
        x = self.input_layer(x)
        x = self.conv1(x,g.edge_index)
        #x = self.conv2(x,g.edge_index)
        cluster_embeddings = []
        for i in range(num_clusters-1):
            cluster_elements = np.nonzero(clusters==i)[0]
            cluster_embeddings.append(self.embeddings_layer(x[cluster_elements].view(-1)))
        
        cluster_elements = np.nonzero(clusters==(num_clusters-1))[0]
        cluster_embeddings.append(self.embeddings_layer(torch.cat((x[cluster_elements].view(-1),
        torch.zeros(self.embedding_dim*self.target_cluster_size-x[cluster_elements].view(-1).shape[0])))))

        cluster_embeddings = [self.class_token(torch.zeros(1).type(torch.LongTensor)).view(-1)] + cluster_embeddings
        cluster_embeddings = torch.stack(cluster_embeddings)
        result = self.transformer(cluster_embeddings.view((1,cluster_embeddings.shape[0],cluster_embeddings.shape[1])))
        return self.to_class(result[0][0].view(1,-1))
