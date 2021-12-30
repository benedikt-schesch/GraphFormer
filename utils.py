from src.models.models import *

def get_optimizer(args,model):
    optimizer = None
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    print("Optimizer: ",optimizer)
    return optimizer

def get_loss(args):
    loss = None
    if args.loss == "crossentropyloss":
        loss = torch.nn.CrossEntropyLoss()
    print("Loss: ",loss)
    return loss

def get_dataset(args):
    dataset = None
    if args.dataset == "fourcycles":
        dataset = FourCycles()
    elif args.dataset == "cycles":
        dataset = Cycles()
    elif args.dataset == "mutag":
        dataset = TUDatasetMutagGen()
    elif args.dataset == "protein":
        dataset = TUDatasetProteinGen()
    elif args.dataset == "node_detection":
        dataset = MarkedNode()
    elif args.dataset == "triangles":
        dataset = Triangles()
    else:
        raise Exception()
    print("Dataset used: ",dataset.__class__.__name__)
    print(f'Number of nodes: {dataset.num_nodes}')
    return dataset

def get_model(args,embedding_dim,num_clases,device):
    if args.model == "GraphFormer":
        return GraphFormer(embedding_dim,num_clases,device,convs=True)
    if args.model == "GraphFormerNoConvs":
        return GraphFormer(embedding_dim,num_clases,device,convs=False)
    if args.model == "RandomGraphFormer":
        return RandomGraphFormer(embedding_dim,num_clases,device)
    if args.model == "ConvAggrBaseline":
        return ConvAggregationBaseline(embedding_dim,num_clases,device)
    if args.model == "TransformerBaseline":
        return TransformerBaseline(embedding_dim,num_clases,device)
