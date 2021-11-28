import torch
from torch_geometric.utils import degree
import numpy as np
import dgl
import networkx as nx
import random
from math import sqrt, pow
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader

# Synthetic datasets

class SymmetrySet:
    def __init__(self):
        self.hidden_units = 0
        self.num_classes = 0
        self.num_features = 0
        self.num_nodes = 0

    def addports(self, data):
        data.ports = torch.zeros(data.num_edges, 1)
        degs = degree(data.edge_index[0], data.num_nodes, dtype=torch.long) # out degree of all nodes
        for n in range(data.num_nodes):
            deg = degs[n]
            ports = np.random.permutation(int(deg))
            for i, neighbor in enumerate(data.edge_index[1][data.edge_index[0]==n]):
                nb = int(neighbor)
                data.ports[torch.logical_and(data.edge_index[0]==n, data.edge_index[1]==nb), 0] = float(ports[i])
        return data

    def makefeatures(self, data):
        data.x = torch.ones((data.num_nodes, 1))
        data.id = torch.tensor(np.random.permutation(np.arange(data.num_nodes))).unsqueeze(1)
        # data.nodes = torch.tensor([data.num_nodes] * data.num_nodes)
        return data

    def makedata(self):
        pass


class FourCycles(SymmetrySet):
    def __init__(self):
        super().__init__()
        self.p = 4
        self.hidden_units = 16 # 10
        self.num_classes = 2
        self.num_features = 1
        self.num_nodes = 4 * self.p
        self.graph_class = True
    def gen_graph(self, p):
        edge_index = None
        for i in range(p):
            e = torch.tensor([[i, p + i, 2 * p + i, 3 * p + i], [2 * p + i, 3 * p + i, i, p + i]], dtype=torch.long)
            if edge_index is None:
                edge_index = e
            else:
                edge_index = torch.cat([edge_index, e], dim=-1)
        top = np.zeros((p * p,))
        perm = np.random.permutation(range(p))
        for i, t in enumerate(perm):
            top[i * p + t] = 1
        bottom = np.zeros((p * p,))
        perm = np.random.permutation(range(p))
        for i, t in enumerate(perm):
            bottom[i * p + t] = 1
        for i, bit in enumerate(top):
            if bit:
                e = torch.tensor([[i // p, p + i % p], [p + i % p, i // p]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
        for i, bit in enumerate(bottom):
            if bit:
                e = torch.tensor([[2 * p + i // p, 3 * p + i % p], [3 * p + i % p, 2 * p + i // p]], dtype=torch.long)
                edge_index = torch.cat([edge_index, e], dim=-1)
        return Data(edge_index=edge_index, num_nodes=self.num_nodes), any(np.logical_and(top, bottom))
    def makedata(self):
        size = 25
        p = self.p
        trues = []
        falses = []
        while len(trues) < size or len(falses) < size:
            data, label = self.gen_graph(p)
            data = self.makefeatures(data)
            data = self.addports(data)
            data.y = label
            if label and len(trues) < size:
                trues.append(data)
            elif not label and len(falses) < size:
                falses.append(data)
        return trues + falses
    
class TUDatasetProteinGen():
    def __init__(self) -> None:
        self.num_nodes = -1
        self.num_classes = 2
        self.num_features = 3
        self.graph_class = True
    
    def makedata(self):
        dataset = TUDataset(root='src/data/TUDataset', name='PROTEINS')
        return dataset


class TUDatasetMutagGen():
    def __init__(self) -> None:
        self.num_nodes = -1
        self.num_classes = 2
        self.num_features = 7
        self.graph_class = True
    
    def makedata(self):
        dataset = TUDataset(root='src/data/TUDataset', name='MUTAG')
        return dataset


class MarkedNode():
    def __init__(self) -> None:
        self.num_nodes = 64
        self.num_edges = 256

        self.num_classes = 2
        self.num_features = 1
        self.num_samples = 1024
        self.num_marked = 64
        self.graph_class = True
    
    def makedata(self):
        dataset = []
        labels = []
        for i in range(self.num_samples):
            graph = dgl.rand_graph(num_nodes=self.num_nodes,num_edges=self.num_edges)
            graph.ndata["x"] = torch.zeros((self.num_nodes,1))
            tagged_nodes = random.sample(list(range(self.num_nodes)),self.num_marked)
            for j in tagged_nodes:            
                graph.ndata["x"][j,0] = int(i>(self.num_samples/2))
            labels.append(int(i>(self.num_samples/2)))
            dataset.append(graph)
        return dataset, torch.tensor(labels)


class Triangles():
    def __init__(self) -> None:
        self.num_nodes = 64
        self.max_num_edges = 256
        self.num_classes = 2
        self.num_features = 1
        self.num_samples = 1024
        self.graph_class = True
    
    def makedata(self):
        dataset = []
        labels = []
        i = 0
        while i < self.num_samples/2:
            graph = nx.gnm_random_graph(self.num_nodes,random.randint(16,self.max_num_edges))
            if sum(nx.triangles(graph).values()) > 30:
                graph = dgl.from_networkx(graph)
                graph.ndata["x"] = torch.zeros((self.num_nodes,1))
                labels.append(1)
                dataset.append(graph)
                i+=1
        i = 0
        while i < self.num_samples/2:
            graph = nx.gnm_random_graph(self.num_nodes,random.randint(16,self.max_num_edges))
            if sum(nx.triangles(graph).values()) == 0:
                graph = dgl.from_networkx(graph)
                graph.ndata["x"] = torch.zeros((self.num_nodes,1))
                labels.append(0)
                dataset.append(graph)
                i+=1
        return dataset, torch.tensor(labels)


class Cycles():
    def __init__(self) -> None:
        self.num_nodes = 64
        self.max_num_edges = 256
        self.num_classes = 2
        self.num_features = 1
        self.num_samples = 1024
        self.graph_class = True
    
    def makedata(self):
        dataset = []
        labels = []
        i = 0
        while i < self.num_samples/2:
            graph = nx.gnm_random_graph(self.num_nodes,random.randint(16,self.max_num_edges))
            if len([i for i in nx.cycle_basis(graph)]) > 1:
                graph = dgl.from_networkx(graph)
                graph.ndata["x"] = torch.zeros((self.num_nodes,1))
                labels.append(1)
                dataset.append(graph)
                i+=1
        i = 0
        while i < self.num_samples/2:
            graph = nx.gnm_random_graph(self.num_nodes,random.randint(16,self.max_num_edges))
            if len([i for i in nx.cycle_basis(graph)]) == 0:
                graph = dgl.from_networkx(graph)
                graph.ndata["x"] = torch.zeros((self.num_nodes,1))
                labels.append(0)
                dataset.append(graph)
                i+=1
        return dataset, torch.tensor(labels)
