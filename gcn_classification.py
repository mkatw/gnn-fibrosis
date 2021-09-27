from argparse import ArgumentParser
from glob import glob
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import distance
from skimage import morphology, measure
import random
import math
import time
import matplotlib.pyplot as plt

import networkx as nx
from torch_geometric.utils import from_networkx
import torch
from torch_geometric.data import DataLoader
from torch.nn import Linear, Sequential, Tanh
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool
import torch_scatter
from torch.nn.utils.rnn import pad_sequence
import pickle
from ignite import metrics


def get_euclidean_distance(u, v):
    edge_wt = round(distance.euclidean(u, v),3)
    return edge_wt


def get_biopsy_core(prediction, background_value=-1, min_tile_number=20):
    # retain only the tiles from the main biopsy core -- remove small bits of tissue
    # set min_tile_number > 1 to remove isolated tiles (and therefore isolated graph nodes)

    tissue_mask = prediction > background_value
    tissue_mask = morphology.remove_small_objects(tissue_mask, min_size=min_tile_number)
    small_bits_mask = (prediction > background_value) ^ tissue_mask
    prediction[small_bits_mask] = background_value
    prediction = np.ma.array(prediction, mask=(prediction == background_value))  # making background transparent
    return prediction


def get_centroids(prediction):
    # find centroids of class 3 tile regions ('red regions')
    red_regions = measure.label(prediction > 2)
    centroids = []
    props = measure.regionprops(red_regions)
    for i, prop in enumerate(props):
        (c_x) = int(round(props[i].centroid[0]))
        (c_y) = int(round(props[i].centroid[1]))
        centroids.append((c_x, c_y))
    return centroids


def FibGraphDataset(root_dir, n_classes):
    
    data_list = []
    prediction_maps = glob(os.path.join(root_dir, '*_prediction.npy'))
    features_maps = [path.replace('_prediction.npy', '_features_map.npy') for path in prediction_maps]
    casewise_labels_df = pd.read_csv('./casewise_labels.csv',
                                index_col='ID')
    casewise_labels = casewise_labels_df.to_dict()

    for p_map, f_map in zip(prediction_maps, features_maps):
    
        slide_name = Path(p_map).stem
        ID = int(slide_name.split('-0')[0].split('_')[-1])  # exctract ID from slide name: adjust to match your data

        prediction = np.load(p_map)
        feature_map = np.load(f_map)

        threshold = -1

        prediction = get_biopsy_core(prediction, background_value=-1, min_tile_number=20)
        centroids = get_centroids(prediction)

        # construct graph: map each tile to node
        G = nx.Graph()
        x_range, y_range = prediction.shape
    
        for centroid in centroids:
            G.add_node(centroid, value=prediction[centroid])
    
        for x in range(x_range-1):
            for y in range(y_range-1):
                if (prediction[x,y] > threshold):
                    G.add_node((x,y), value=prediction[x,y])
                if (prediction[x,y] > threshold) and (prediction[x,y+1] > threshold):
                    G.add_edge((x,y),(x,y+1))
                if (prediction[x,y] > threshold) and (prediction[x+1,y] > threshold):
                    G.add_edge((x,y),(x+1,y))
                
        for isolate in list(nx.isolates(G)):
            G.remove_node(isolate)  # this is because isolated nodes crash tessellation
            centroids.remove(isolate)

        # construct tessellated graph
        try:

            tesselation = nx.voronoi_cells(G, centroids)

            H = nx.Graph(tesselation)

            for node in list(H.nodes()):
                if node == 'unreachable':  # unreachable nodes crash plotting
                    H.remove_node(node)

            for isolate in list(nx.isolates(H)):
                H.remove_node(isolate)

            pos = dict(zip(H.nodes(), H.nodes()))  # map node names to coordinates
            nx.draw(H, pos, node_size=1)

            # assigning weights and attributes to H
            for edge in list(H.edges(data=True)):
                weight = get_euclidean_distance(edge[0], edge[1])
                H[edge[0]][edge[1]]['weight'] = weight.astype(np.float64)
                H.nodes[edge[1]]['x'] = feature_map[:, edge[1][0], edge[1][1]].astype(np.float64)

            case_class = torch.tensor(np.array(int(casewise_labels['fib'][ID])))

            # merging classes options
            if n_classes == 4:  # 0 v 1 v 2 v 3 classification
                pass
            if n_classes == 3:  # (0, 1) v 2 v 3
                case_class[case_class < 2] = 0
                case_class[case_class == 2] = 1
                case_class[case_class == 3] = 2
            elif n_classes == 2:  # (0, 1, 2) v 3
                case_class[case_class < 3] = 0
                case_class[case_class == 3] = 1

            data_graph = from_networkx(H)

        except:
            print(f'Case {ID} failed graph construction.')
            pass

        data_graph['name'] = ID
        data_graph['y'] = case_class.long()
        data_graph['pos'] = pos
        data_list.append(data_graph)

    return data_list


def print_dataset_characteristics(dataset):

    print()
    print(f'Dataset:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    #print(f'Number of features: {dataset.num_features}')
    #print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
    print('=============================================================')

    # Gather some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Contains isolated nodes: {data.contains_isolated_nodes()}')
    print(f'Contains self-loops: {data.contains_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')


def dataset_loaders(dataset):

    random.seed(4)
    random.shuffle(dataset)
    s = [0, 1, 2, 3]  # metavir stages

    for stage in s:
        s[stage] = [data for data in dataset if data.y == stage]
        print(f'Number of stage {stage} cases: {len(s[stage])}')

    train_dataset = []
    test_dataset = []
    val_dataset = []

    # 80-10-10 split
    for grade in range(4):
        train_dataset.extend(s[grade][0:int(math.floor(len(s[grade]) * 0.8))])
        val_dataset.extend(s[grade][int(math.floor(len(s[grade]) * 0.8)):int(math.floor(len(s[grade]) * 0.9))])
        test_dataset.extend(s[grade][int(math.floor(len(s[grade]) * 0.9)):])

    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'0: {len(s[0][0:int(math.floor(len(s[0]) * 0.8))])} 1: {len(s[1][0:int(math.floor(len(s[1]) * 0.8))])} 2: {len(s[2][0:int(math.floor(len(s[2]) * 0.8))])} 3: {len(s[3][0:int(math.floor(len(s[3]) * 0.8))])}')
    print(f'Number of validation graphs: {len(val_dataset)}')
    print(f'0: {len(s[0][int(math.floor(len(s[0]) * 0.8)):int(math.floor(len(s[0]) * 0.9))])} 1: {len(s[1][int(math.floor(len(s[1]) * 0.8)):int(math.floor(len(s[1]) * 0.9))])} 2: {len(s[2][int(math.floor(len(s[2]) * 0.8)):int(math.floor(len(s[2]) * 0.9))])} 3: {len(s[3][int(math.floor(len(s[3]) * 0.8)):int(math.floor(len(s[3]) * 0.9))])}')
    print(f'Number of test graphs: {len(test_dataset)}')
    print(f'0: {len(s[0][int(math.floor(len(s[0]) * 0.9)):])} 1: {len(s[1][int(math.floor(len(s[1]) * 0.9)):])} 2: {len(s[2][int(math.floor(len(s[2]) * 0.9)):])} 3: {len(s[3][int(math.floor(len(s[3]) * 0.9)):])}')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for step, data in enumerate(train_loader):
        print(f'Step {step + 1}:')
        print('=======')
        print(f'Number of graphs in the current batch: {data.num_graphs}')
        print(data)
        print()

    #for data in val_loader:
        #print('Val dataset:')
        #print(f'Case: {data.name}, stage: {data.y}')

    return train_loader, test_loader, val_loader


class Attention(torch.nn.Module):

    def __init__(self, n_features, n_classes=1):
        super(Attention, self).__init__()

        self.n_classes = n_classes
        self.U = torch.nn.Linear(n_features, n_classes, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.classifier = torch.nn.Conv1d(n_classes, n_classes, n_features,
                                    groups=n_classes)

    def forward(self, x):
        a = self.softmax(self.U(x).transpose(-2, -1))
        x_embed = a.matmul(x)
        x_embed = F.dropout(x_embed, p=0.2, training=self.training)
        y_logits = self.classifier(x_embed).squeeze(-1).squeeze(-1)
        return y_logits, a


class GCN(torch.nn.Module):

    def __init__(self, hidden_channels, num_node_features, num_classes, GCN_layer, attention=False):
        super(GCN, self).__init__()
        if GCN_layer == GINConv:
            self.conv1 = GCN_layer(Sequential(Linear(num_node_features, hidden_channels), Tanh()))
            self.conv2 = GCN_layer(Sequential(Linear(hidden_channels, hidden_channels), Tanh()))
            self.conv3 = GCN_layer(Sequential(Linear(hidden_channels, hidden_channels), Tanh()))

        else:
            self.conv1 = GCN_layer(num_node_features, hidden_channels)
            self.conv2 = GCN_layer(hidden_channels, hidden_channels)
            self.conv3 = GCN_layer(hidden_channels, hidden_channels)
        self.attention = attention
        if attention:
            self.attn = Attention(hidden_channels, num_classes)
        else:
            self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        if self.attention:
            x = torch.tanh(x)
            _, counts = torch.unique(batch, return_counts=True)
            x = pad_sequence(torch.split(x, counts.tolist(), 0), batch_first=True)
            x, _ = self.attn(x)

        else:

            # 2. Readout layer
            x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
            #x = global_max_pool(x, batch)

            # 3. Apply a final classifier
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.lin(x)
        
        return x


class EarlyStopping:
    def __init__(self, patience):
        self.best_score = 0
        self.epochs_elapsed = 0
        self.patience = patience

    def update(self, score):
        if score > self.best_score:
            self.best_score = score
            self.epochs_elapsed = 0
            return True
        else:
            self.epochs_elapsed += 1
            return False

    def stop(self):
        return True if self.epochs_elapsed >= self.patience else False


def train(model, optimizer, criterion, train_loader):

    model.train()
    loss_epoch = 0

    for i, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
        x, y, edge_index, batch = data.x, data.y, data.edge_index, data.batch
        x = x.cuda()
        y = y.cuda()
        edge_index = edge_index.cuda()
        batch = batch.cuda()

        out = model(x.double(), edge_index, batch)  # Perform a single forward pass.
        loss = criterion(out, y.long())  # Compute the loss.
        loss.backward()  # Derive gradients.
        loss_epoch += loss.item()
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

    return loss_epoch/(i+1)


class ROC_AUC(metrics.EpochMetric):
    def __init__(self, output_transform=lambda x: x, check_compute_fn=False, **kwargs):
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            raise RuntimeError("This module requires sklearn to be installed.")
        self.args = kwargs

        def roc_auc_compute_fn(y_preds, y_targets):
            y_true = y_targets.numpy()
            y_pred = y_preds.numpy()
            return roc_auc_score(y_true, y_pred, **self.args)

        super(ROC_AUC, self).__init__(roc_auc_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn)


class PR_AUC(metrics.EpochMetric):
    def __init__(self, output_transform=lambda x: x, check_compute_fn=False, **kwargs):
        try:
            from sklearn.metrics import precision_recall_curve, auc
        except ImportError:
            raise RuntimeError("This module requires sklearn to be installed.")
        self.args = kwargs

        def pr_auc_compute_fn(y_preds, y_targets):
            y_true = y_targets.numpy()
            y_pred = y_preds.numpy()
            precision, recall, _ = precision_recall_curve(y_true, y_pred, **self.args)
            return auc(recall, precision)

        super(PR_AUC, self).__init__(pr_auc_compute_fn, output_transform=output_transform, check_compute_fn=check_compute_fn)


def test(model, test_loader, criterion, metrics):
    model.eval()
    loss_epoch = 0
    softmax = torch.nn.Softmax(1)

    for metric in metrics.values():
        metric.reset()

    for i, data in enumerate(test_loader):  # Iterate in batches over the training/test dataset.
        x, y, edge_index, batch = data.x, data.y, data.edge_index, data.batch
        x = x.cuda()
        y = y.cuda()
        edge_index = edge_index.cuda()
        batch = batch.cuda()
        out = softmax(model(x, edge_index, batch))
        loss = criterion(out, y.long())  # Compute the loss.
        loss_epoch += loss.item()

        for metric in metrics.values():
            metric.update(metric._output_transform((out, y)))

    metrics = {metric_name: metric.compute() for metric_name, metric in metrics.items()}

    return metrics, loss_epoch/(i+1)


def main(args):

    dataset_path = f'./res_net/gcn_dataset_{args.n_classes}.npy'

    if args.create_dataset:
        dataset = FibGraphDataset(
            './res_net_prediction_maps/', args.n_classes)
        with open(dataset_path, "wb") as fp:
            pickle.dump(dataset, fp)

    else:
        with open(dataset_path, "rb") as fp:
            dataset = pickle.load(fp)

    print_dataset_characteristics(dataset)
    train_loader, test_loader, val_loader = dataset_loaders(dataset)
    num_classes = args.n_classes
    num_node_features = 512

    gnn_layers = {
        'GCN': GCNConv,
        'GIN': GINConv,
        'GAT': GATConv}

    model = GCN(64, num_node_features, num_classes, gnn_layers[args.layers], args.attention)
    model.cuda()
    print(model)

    model_dir = args.saved_models_dir / Path(model.__class__.__name__ + '_' + time.strftime('%Y-%m-%d_%H:%M:%S'))
    os.mkdir(model_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    model = model.double()
    early_stopping = EarlyStopping(args.patience)

    cm = metrics.ConfusionMatrix(num_classes)
    eval_metrics = {'confusion_matrix': cm, 'dice': metrics.DiceCoefficient(cm)}#,
    #                'ROC': ROC_AUC(multi_class='ovo', average='weighted')}
    log = open(model_dir / Path('metrics.txt'), 'w')

    training_loss_vector = []
    test_loss_vector = []

    fig = plt.figure()
    fig.set_size_inches(18, 7)

    for epoch in range(1, args.n_epochs):
        train_loss = train(model, optimizer, criterion, train_loader)
        train_metrics, _ = test(model, train_loader, criterion, eval_metrics)
        training_loss_vector.append(train_loss)
        test_metrics, test_loss = test(model, val_loader, criterion, eval_metrics)
        test_loss_vector.append(test_loss)
        is_best = early_stopping.update(test_metrics["dice"].mean())

        print(f'Epoch: {epoch:03d}, Train Dice: {train_metrics["dice"]},  Test Dice: {test_metrics["dice"]}')

        if is_best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, model_dir / Path('model_best_dice.pt'))

            print(f'Epoch: {epoch:03d}, Train Dice: {train_metrics["dice"]},\n  Test Dice: {test_metrics["dice"]}\n',
                  file=log)

        if early_stopping.stop():
            print('early stopping')
            val_metrics, _ = test(model, test_loader, criterion, eval_metrics)
            print(f'Epoch: {epoch:03d}, Validation Dice: {val_metrics["dice"]},\n Confusion Matrix: {val_metrics["confusion_matrix"]}', file=log)
            break

        plt.plot(training_loss_vector)
        plt.plot(test_loss_vector)
        plt.savefig("val_loss.png", dpi=None, facecolor='w', edgecolor='w',
                    orientation='portrait', format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1, metadata=None)


parser = ArgumentParser(description='Train fibrosis GCN')
parser.add_argument('-o', dest='saved_models_dir', required=True, type=Path)
parser.add_argument('--c', dest='n_classes', type=int, default=4)
parser.add_argument('--patience', dest='patience', type=int, default=100)
parser.add_argument('--epochs', dest='n_epochs', type=int, default=400)
parser.add_argument('--layers', default='GCN', choices=['GCN', 'GIN', 'GAT'])
parser.add_argument('--create_dataset', default=False, action='store_true')
parser.add_argument('--attention', default=False, action='store_true')

args = parser.parse_args()

main(args)
