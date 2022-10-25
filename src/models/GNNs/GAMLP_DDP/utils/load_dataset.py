import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from .heteo_data import load_data, read_relation_subsets, gen_rel_subset_feature, preprocess_features
import torch.nn.functional as F
import gc
import os
from utils.function.np_utils import *
from tqdm import tqdm


def prepare_label_emb(label_num_hops, g, labels, n_classes, args, train_idx, valid_idx, test_idx):
    print(n_classes)
    print(labels.shape[0])
    if 'ind' in args.dataset:
        labels = labels[g.ndata['_ID']]
    if labels.ndim == 1:
        y = np.zeros(shape=(labels.shape[0], int(n_classes)))
        y[train_idx] = F.one_hot(labels[train_idx].to(
            torch.long), num_classes=n_classes).float().squeeze(1)
        y = torch.Tensor(y)
    elif labels.ndim == 2:
        y = labels
    else:
        ValueError(f'Labels dimension wrong')

    del labels
    gc.collect()
    for hop in range(label_num_hops):
        y = neighbor_average_labels(g, y.to(torch.float))
        gc.collect()
    res = y
    save_memmap(res, os.path.join('temp', args.processed_label), np.float16)
    return  # res #torch.cat([res[train_idx], res[valid_idx], res[test_idx]], dim=0)


def neighbor_average_labels(g, feat):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged labels")
    g.ndata["f"] = feat
    g.update_all(fn.copy_u("f", "msg"),
                 fn.mean("msg", "f"))
    feat = g.ndata.pop('f')

    '''if args.dataset == "ogbn-mag":
        # For MAG dataset, only return features for target node types (i.e.
        # paper nodes)

        # error
        target_mask = g.ndata['target_mask']
        target_ids = g.ndata[dgl.NID][target_mask]
        num_target = target_mask.sum().item()
        new_res = []
        for x in res:
            feat = torch.zeros((num_target,) + x.shape[1:],
                               dtype=x.dtype, device=x.device)
            feat[target_ids] = x[target_mask]
            new_res.append(feat)
        res = new_res'''

    return feat


def neighbor_average_features(g, features, args):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats")
    if 'ind' in args.dataset:
        features = features[g.ndata['_ID']]
    g.ndata["feat_0"] = features
    save_memmap(g.ndata['feat_0'], os.path.join('temp', args.processed_feat(0)), np.float16)
    for hop in tqdm(range(1, args.num_hops + 1)):
        if hop > 1:
            g.ndata.pop(f'feat_{hop - 2}')
        g.update_all(fn.copy_u(f"feat_{hop - 1}", "msg"),
                     fn.mean("msg", f"feat_{hop}"))
        print('start mmp')
        save_memmap(g.ndata[f"feat_{hop}"], os.path.join('temp', args.processed_feat(hop)), np.float16)
    del features
    return


def batched_acc(labels, pred):
    # testing accuracy for single label multi-class prediction
    return (torch.argmax(pred, dim=1) == labels,)


def get_ogb_evaluator(dataset):
    """
    Get evaluator from Open Graph Benchmark based on dataset
    """
    #    if dataset=='ogbn-mag':
    #        return batched_acc
    #    else:
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval({
        "y_true": labels.view(-1, 1),
        "y_pred": preds.view(-1, 1),
    })["acc"]


def load_dataset(name, device, args):
    """
    Load dataset and move graph and features to device
    """
    '''if name not in ["ogbn-products", "ogbn-arxiv","ogbn-mag"]:
        raise RuntimeError("Dataset {} is not supported".format(name))'''
    if name not in ["ogbn-products", "ogbn-mag", "ogbn-papers100M"]:
        raise RuntimeError("Dataset {} is not supported".format(name))
    dataset = DglNodePropPredDataset(name=name, root=args.root)
    splitted_idx = dataset.get_idx_split()

    if name == "ogbn-products":
        train_nid = splitted_idx["train"]
        val_nid = splitted_idx["valid"]
        test_nid = splitted_idx["test"]
        g, labels = dataset[0]
        g.ndata["labels"] = labels
        g.ndata['feat'] = g.ndata['feat'].float()
        n_classes = dataset.num_classes
        labels = labels.squeeze()
        evaluator = get_ogb_evaluator(name)
    elif name == "ogbn-mag":
        data = load_data(device, args)
        g, labels, n_classes, train_nid, val_nid, test_nid = data
        evaluator = get_ogb_evaluator(name)
    elif name == "ogbn-papers100M":
        train_nid = splitted_idx["train"]
        val_nid = splitted_idx["valid"]
        test_nid = splitted_idx["test"]
        g, labels = dataset[0]
        n_classes = dataset.num_classes
        labels = labels.squeeze()
        evaluator = get_ogb_evaluator(name)
    print(f"# Nodes: {g.number_of_nodes()}\n"
          f"# Edges: {g.number_of_edges()}\n"
          f"# Train: {len(train_nid)}\n"
          f"# Val: {len(val_nid)}\n"
          f"# Test: {len(test_nid)}\n"
          f"# Classes: {n_classes}\n")

    return g, labels, n_classes, train_nid, val_nid, test_nid, evaluator


def prepare_data(device, args, teacher_probs):
    """
    Load dataset and compute neighbor-averaged node features used by SIGN model
    """

    data = load_dataset(args.dataset, device, args)

    g, labels, n_classes, train_nid, val_nid, test_nid, evaluator = data
    if 'ind' in args.dataset:
        labels = labels[g.ndata['_ID']]
    if args.dataset == 'ogbn-products':
        feats = neighbor_average_features(g, args)
        in_feats = feats[0].shape[1]
    elif args.dataset == 'ogbn-mag':
        rel_subsets = read_relation_subsets(args.use_relation_subsets)

        with torch.no_grad():
            feats = preprocess_features(g, rel_subsets, args, device)
            print("Done preprocessing")
        _, num_feats, in_feats = feats[0].shape
    elif args.dataset == 'ogbn-papers100M':
        g = dgl.add_reverse_edges(g, copy_ndata=True)
        feat = g.ndata.pop('feat')
    gc.collect()
    label_emb = None
    if args.use_rlu:
        label_emb = prepare_label_emb(args, g, labels, n_classes, train_nid, val_nid, test_nid, teacher_probs)
    # move to device
    if args.dataset == 'ogbn-papers100M':

        feats = []
        for i in range(args.num_hops + 1):
            feats.append(torch.load(f"/data2/zwt/ogbn_papers100M/feat/papers100m_feat_{i}.pt"))
        in_feats = feats[0].shape[1]
        '''
        g.ndata['feat']=feat
        feats=neighbor_average_features(g,args)
        in_feats=feats[0].shape[1]
        
        for i, x in enumerate(feats):
            feats[i] = torch.cat((x[train_nid], x[val_nid], x[test_nid]), dim=0)
        '''
    else:
        for i, x in enumerate(feats):
            feats[i] = torch.cat((x[train_nid], x[val_nid], x[test_nid]), dim=0)
    train_nid = train_nid.to(device)
    val_nid = val_nid.to(device)
    test_nid = test_nid.to(device)
    labels = labels.to(device).to(torch.long)
    return feats, torch.cat([labels[train_nid], labels[val_nid], labels[test_nid]]), in_feats, n_classes, \
           train_nid, val_nid, test_nid, evaluator, label_emb
