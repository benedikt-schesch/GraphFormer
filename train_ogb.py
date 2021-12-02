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
from sklearn.model_selection import train_test_split
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl
import matplotlib.pyplot as plt
from ogb.graphproppred import Evaluator
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader

def evaluate(model,loader,loss,device,evaluator):
    num_correct = 0
    total_loss = 0
    num_items = 0
    y_pred = []
    y_true = []
    for x, label in loader:
        x, label = dgl.unbatch(x), label.to(device)

        y = []
        for g in x:
            g_gpu = g.to(device)
            y.append(model(g_gpu,g))
        y = torch.cat(y)

        predictions = y.argmax(dim=1, keepdim=True).squeeze()
        num_correct += (predictions == label.view(-1)).sum().item()
        total_loss += loss(y,label.view(-1)).item()*len(y)
        num_items += len(y)
        y_true.append(label)
        y_pred.append(predictions)

    y_true = torch.cat((y_true)).view(-1,1)
    y_pred = torch.cat((y_pred)).view(-1,1)
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    result_dict = evaluator.eval(input_dict)
    return result_dict["rocauc"], total_loss/num_items

def main(args):
    print(args, flush=True)
    experiement_id = str(time.time())[:10]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ",device)
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    dataset = DglGraphPropPredDataset(name = args.dataset)
    evaluator = Evaluator(name = args.dataset)
    dataset.graphs = [dgl.add_reverse_edges(graph) for graph in dataset.graphs]
    dataset.graphs = [dgl.add_self_loop(graph) for graph in dataset.graphs]
    for graph in dataset.graphs:
        graph.ndata["x"] = graph.ndata["feat"].type(torch.float)
    print("Average number of nodes: ",sum([graph.num_nodes() for graph in dataset.graphs])/len(dataset.graphs))
    
    split_idx = dataset.get_idx_split()
    train_idx = torch.cat((split_idx["test"],split_idx["train"]))
    
    test_idx = split_idx["valid"]
    train_loader = DataLoader(dataset[train_idx], sampler=ImbalancedDatasetSampler(dataset[train_idx],
        callback_get_label = lambda x: torch.cat([i[1] for i in x])),batch_size=32,collate_fn=collate_dgl)
    test_loader = DataLoader(dataset[test_idx],batch_size=32,collate_fn=collate_dgl)
    
    model = GraphTransformer(args.embedding_dim,args.num_agents,args.num_hops,
                int(dataset.num_classes),dataset[0][0].ndata["feat"].shape[1],device)
    model.to(device)
    optimizer = get_optimizer(args,model)
    loss_func = get_loss(args)

    train_infos = pd.DataFrame(columns=['Epoch','Training accuracy','Testing ROC','Training loss','Testing loss'])

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
                    g_gpu = g.to(device)
                    y.append(model(g_gpu, g))
                #make_dot(y[0], params=dict(list(model.named_parameters()))).save("rnn_torchviz")
                
                y = torch.cat(y)
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
            roc, test_loss = evaluate(model,test_loader,loss_func,device,evaluator)

            train_infos = train_infos.append({"Epoch":epoch,
                            "Training accuracy":train_accuracy,
                            "Testing ROC": roc,
                            "Training loss":train_loss,
                            "Testing loss":test_loss},ignore_index=True)


    #Create accuracy and loss plots
    ax = train_infos[['Training accuracy']].plot(xlim=[0,args.epochs-1],ylim=[0,1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    ax.get_figure().savefig('figs/acc_experiment_ogb_'+str(experiement_id))

    ax = train_infos[['Training loss','Testing loss']].plot(xlim=[0,args.epochs-1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    ax.get_figure().savefig('figs/loss_experiment_ogb_'+str(experiement_id))


    ax = train_infos[['Testing ROC']].plot(xlim=[0,args.epochs-1],ylim=[0,1])
    plt.xlabel('Epoch')
    plt.ylabel('ROC')
    ax.get_figure().savefig('figs/roc_experiment_ogb_'+str(experiement_id))

    test_roc, test_loss = evaluate(model,test_loader,loss_func,device,evaluator)

    return train_accuracy, test_roc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=int, default=1, help="Options are: 0, 1")
    parser.add_argument('--dataset', type=str, default='ogbg-ppa', help="Options are: ogbg-ppa")
    parser.add_argument('--optimizer', type=str, default='adam', help="Options are: adam, sgd")
    parser.add_argument('--loss', type=str, default='crossentropyloss', help="Options are: crossentropyloss")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--epochs', type=int, default=0)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()
    main(args)