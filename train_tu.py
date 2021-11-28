import argparse
import torch
import dgl
import time
import torch_geometric
from utils import *
from tqdm import tqdm
import pandas as pd
from torchviz import make_dot
from src.models.GraphTransformer import *
from dgl.data.tu import TUDataset
from sklearn.model_selection import train_test_split
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
import matplotlib.pyplot as plt
from ogb.graphproppred import Evaluator
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader

def evaluate(model,loader,loss,device):
    num_correct = 0
    total_loss = 0
    num_items = 0
    y_pred = []
    y_true = []
    for x, label in loader:
        x, label = dgl.unbatch(x), label.to(device)

        y = []
        for g in x:
            g = g.to(device)
            y.append(model(g))
        y = torch.stack(y)

        predictions = y.argmax(dim=1, keepdim=True).squeeze()
        num_correct += (predictions == label.view(-1)).sum().item()
        total_loss += loss(y,label.view(-1)).item()*len(y)
        num_items += len(y)
        y_true.append(label)
        y_pred.append(predictions)

    y_true = torch.cat((y_true)).view(-1,1)
    y_pred = torch.cat((y_pred)).view(-1,1)
    return ((y_pred == y_true).sum()/len(y_pred)).item(), total_loss/num_items

def main(args):
    print(args, flush=True)
    experiement_id = str(time.time())[:10]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ",device)
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset = TUDataset(name=args.dataset)
    print("Average number of nodes: ",sum([graph[0].num_nodes() for graph in dataset])/len(dataset))
    print("Data Imbalance: ",sum(dataset.graph_labels).item()/len(dataset))
    
    train_idx, test_idx = train_test_split(range(len(dataset)),test_size=0.25)
    
    train_loader = DataLoader([dataset[i] for i in train_idx], batch_size=args.batch_size,collate_fn=collate_dgl)
    test_loader = DataLoader([dataset[i] for i in test_idx],batch_size=args.batch_size,collate_fn=collate_dgl)
    
    model = GraphTransformer(args.embedding_dim,num_features=-1,num_clases=dataset.num_labels[0],device=device)
    model.to(device)
    optimizer = get_optimizer(args,model)
    loss_func = get_loss(args)

    train_infos = pd.DataFrame(columns=['Epoch','Training accuracy','Testing Accuracy','Training loss','Testing loss'])

    for epoch in range(args.epochs):
        with tqdm(train_loader, unit="batch",disable=(args.verbose==0)) as tepoch:
            total_correct = 0
            total_loss = 0
            num_items = 0
            for x, label in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                x, label = dgl.unbatch(x), label.to(device)

                optimizer.zero_grad()
                y = []
                for g in x:
                    g = g.to(device)
                    y.append(model(g))
                #make_dot(y[0], params=dict(list(model.named_parameters()))).save("rnn_torchviz")
                
                y = torch.stack(y)
                #if y.isnan().any():
                #    continue

                num_items += len(y)
                predictions = y.argmax(dim=1, keepdim=True).squeeze()
                loss = loss_func(y, label.view(-1))
                correct = (predictions == label.view(-1)).sum().item()
                #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

                loss.backward()
                optimizer.step()
                total_correct += correct
                total_loss += loss.item()*args.batch_size
                tepoch.set_postfix(loss=total_loss/num_items, accuracy=total_correct/num_items)
            train_accuracy = total_correct/num_items
            train_loss = total_loss/num_items
            acc, test_loss = evaluate(model,test_loader,loss_func,device)

            train_infos = train_infos.append({"Epoch":epoch,
                            "Training accuracy":train_accuracy,
                            "Testing accuracy": acc,
                            "Training loss":train_loss,
                            "Testing loss":test_loss},ignore_index=True)


    #Create accuracy and loss plots
    ax = train_infos[['Training accuracy']].plot(xlim=[0,args.epochs-1],ylim=[0,1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    ax.get_figure().savefig('figs/acc_experiment_tu_'+str(experiement_id))

    ax = train_infos[['Training loss','Testing loss']].plot(xlim=[0,args.epochs-1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax.get_figure().savefig('figs/loss_experiment_tu_'+str(experiement_id))


    ax = train_infos[['Testing Accuracy']].plot(xlim=[0,args.epochs-1],ylim=[0,1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    ax.get_figure().savefig('figs/test_acc_experiment_ogb_'+str(experiement_id))

    test_roc, test_loss = evaluate(model,test_loader,loss_func,device)

    return train_accuracy, test_roc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=int, default=1, help="Options are: 0, 1")
    parser.add_argument('--dataset', type=str, default='DD', help="Options are: DD,FIRSTMM_DB,REDDIT_BINARY")
    parser.add_argument('--optimizer', type=str, default='adam', help="Options are: adam, sgd")
    parser.add_argument('--loss', type=str, default='crossentropyloss', help="Options are: crossentropyloss")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()
    main(args)