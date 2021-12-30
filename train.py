import argparse
import torch
import time
import torch_geometric
from utils import *
from tqdm import tqdm
import os
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import torch_geometric.utils
from src.models.models import *
import torch_geometric
import torch_geometric.transforms
from torch_geometric.datasets import TUDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from itertools import groupby


def evaluate(model,dataset2,loader,loss,device):
    model.eval()
    num_correct = 0
    total_loss = 0
    num_items = 0
    y_pred = []
    y_true = []
    for idx in loader:
        x, adj = dataset2[idx]
        x = x.to(device)
        label = x.y
        
        y = model(x, adj)

        predictions = y.argmax(dim=1, keepdim=True).squeeze()
        num_correct += (predictions == label.view(-1)).sum().item()
        total_loss += loss(y,label.view(-1)).item()*len(y)
        num_items += len(y)
        y_true.append(label)
        y_pred.append(predictions)

    y_true = torch.cat((y_true)).view(-1,1)
    y_pred = torch.stack((y_pred)).view(-1,1)
    return ((y_pred == y_true).sum()/len(y_pred)).item(), total_loss/num_items

def main(args):
    print(args, flush=True)
    out_folder = args.output_folder + args.experiment_id +"/"
    os.mkdir(out_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ",device)
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset = TUDataset(name=args.dataset,root="dataset")
    print("Average number of nodes: ",sum([graph.num_nodes for graph in dataset])/len(dataset))
    print("Data Imbalance: ",(sum(dataset.data.y)*1./len(dataset)).item())
    
    def f(x):
        x.edge_index = torch_geometric.utils.to_undirected(torch_geometric.utils.add_self_loops(x.edge_index)[0])
        edges = x.edge_index.transpose(1,0).tolist()
        adj = {k: [v[1] for v in g] for k, g in groupby(sorted(edges), lambda e: e[0])}
        return x, adj
    
    dataset2 = list(map(f,dataset))
    skf = StratifiedKFold(n_splits=args.num_splits, shuffle = True, random_state = 42)
    test_accs = []
    for train_idx,test_idx in skf.split(range(len(dataset)),[i.y.item() for i in dataset]):
        test_accs.append([])
        train_loader = DataLoader(list(train_idx), batch_size=args.batch_size,shuffle=True)
        test_loader = DataLoader(list(test_idx),batch_size=args.batch_size)
    
        model = get_model(args,embedding_dim=args.embedding_dim,num_clases=dataset.num_classes,device=device)
        model.to(device)
        optimizer = get_optimizer(args,model)
        loss_func = get_loss(args)
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20)

        train_infos = pd.DataFrame(columns=['Epoch','Training accuracy','Testing Accuracy','Training loss','Testing loss'])
        for epoch in range(args.epochs):
            with tqdm(train_loader, unit="batch",disable=(args.verbose==0)) as tepoch:
                model.train()
                total_correct = 0
                total_loss = 0
                num_items = 0
                for idx in tepoch:
                    tepoch.set_description(f"Epoch {epoch}")
                    x, adj = dataset2[idx]
                    x = x.to(device)
                    label = x.y
                    
                    optimizer.zero_grad()
                    y = model(x,adj)

                    num_items += len(y)
                    predictions = y.argmax(dim=1, keepdim=True).squeeze()
                    loss = loss_func(y, label.view(-1))
                    loss.backward()
                    correct = (predictions == label.view(-1)).sum().item()

                    optimizer.step()
                    total_correct += correct
                    total_loss += loss.item()*args.batch_size
                    tepoch.set_postfix(loss=total_loss/num_items, accuracy=total_correct/num_items)
                scheduler1.step()
                train_accuracy = total_correct/num_items
                train_loss = total_loss/num_items
                test_accuracy, test_loss = evaluate(model,dataset2,test_loader,loss_func,device)
                test_accs[-1].append(test_accuracy)
                train_infos = train_infos.append({"Epoch":epoch,
                                "Training accuracy":train_accuracy,
                                "Testing accuracy": test_accuracy,
                                "Training loss":train_loss,
                                "Testing loss":test_loss},ignore_index=True)

                #Create accuracy and loss plots
                ax = train_infos[['Training accuracy','Testing accuracy']].plot(xlim=[0,args.epochs-1],ylim=[0,1])
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                ax.get_figure().savefig(out_folder+"Accuracy.png")
                plt.close()

                ax = train_infos[['Training loss','Testing loss']].plot(xlim=[0,args.epochs-1])
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                ax.get_figure().savefig(out_folder+"Loss.png")
                plt.close()

                train_infos.to_csv(out_folder+"train_infos.csv")
    test_accs = np.array(test_accs)
    max_arg = np.argmax(np.mean(test_accs,0))
    print("Max Acuracy: ",np.mean(test_accs,0)[max_arg])
    print("Std: ",np.std(test_accs[:,max_arg]))
    return train_accuracy, test_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=int, default=1, help="Options are: 0, 1")
    parser.add_argument('--model', type=str, default='GraphFormer', help="Options are: GraphFormer,GraphFormerNoConvs,RandomGraphFormer,ConvAggrBaseline,TransformerBaseline")
    parser.add_argument('--dataset', type=str, default='DD', help="Options are: DD,FIRSTMM_DB,REDDIT-BINARY")
    parser.add_argument('--optimizer', type=str, default='adam', help="Options are: adam, sgd")
    parser.add_argument('--loss', type=str, default='crossentropyloss', help="Options are: crossentropyloss")
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--num_splits', type=int, default=10)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output_folder', type=str, default="figs/")
    parser.add_argument('--experiment_id', type=str, default=str(time.time()))


    args = parser.parse_args()
    main(args)