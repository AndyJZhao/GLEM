#the entire file is apdated from https://github.com/facebookresearch/NARS/blob/main/data.py
import os
import numpy as np
import torch
import dgl
import dgl.function as fn


###############################################################################
# Loading Relation Subsets
###############################################################################

def read_relation_subsets(fname):
    print("Reading Relation Subsets:")
    rel_subsets = []
    with open(fname) as f:
        for line in f:
            relations = line.strip().split(',')
            rel_subsets.append(relations)
            print(relations)
    return rel_subsets


###############################################################################
# Generate multi-hop neighbor-averaged feature for each relation subset
###############################################################################

def gen_rel_subset_feature(g, rel_subset, args, device):
    """
    Build relation subgraph given relation subset and generate multi-hop
    neighbor-averaged feature on this subgraph
    """
    device = "cpu"
    new_edges = {}
    ntypes = set()
    for etype in rel_subset:
        stype, _, dtype = g.to_canonical_etype(etype)
        src, dst = g.all_edges(etype=etype)
        src = src.numpy()
        dst = dst.numpy()
        new_edges[(stype, etype, dtype)] = (src, dst)
        new_edges[(dtype, etype + "_r", stype)] = (dst, src)
        ntypes.add(stype)
        ntypes.add(dtype)
    new_g = dgl.heterograph(new_edges)

    # set node feature and calc deg
    for ntype in ntypes:
        num_nodes = new_g.number_of_nodes(ntype)
        if num_nodes < g.nodes[ntype].data["feat"].shape[0]:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"][:num_nodes, :]
        else:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"]
        deg = 0
        for etype in new_g.etypes:
            _, _, dtype = new_g.to_canonical_etype(etype)
            if ntype == dtype:
                deg = deg + new_g.in_degrees(etype=etype)
        norm = 1.0 / deg.float()
        norm[torch.isinf(norm)] = 0
        new_g.nodes[ntype].data["norm"] = norm.view(-1, 1).to(device)

    res = []

    # compute k-hop feature
    for hop in range(1, args.num_hops + 1):
        ntype2feat = {}
        for etype in new_g.etypes:
            stype, _, dtype = new_g.to_canonical_etype(etype)
            new_g[etype].update_all(fn.copy_u(f'hop_{hop-1}', 'm'), fn.sum('m', 'new_feat'))
            new_feat = new_g.nodes[dtype].data.pop("new_feat")
            assert("new_feat" not in new_g.nodes[stype].data)
            if dtype in ntype2feat:
                ntype2feat[dtype] += new_feat
            else:
                ntype2feat[dtype] = new_feat
        for ntype in new_g.ntypes:
            assert ntype in ntype2feat  # because subgraph is not directional
            feat_dict = new_g.nodes[ntype].data
            old_feat = feat_dict.pop(f"hop_{hop-1}")
            if ntype == "paper":
                res.append(old_feat.cpu())
            feat_dict[f"hop_{hop}"] = ntype2feat.pop(ntype).mul_(feat_dict["norm"])

    res.append(new_g.nodes["paper"].data.pop(f"hop_{args.num_hops}").cpu())
    return res


###############################################################################
# Dataset (ACM, MAG, OAG) loading
###############################################################################


def load_data(device, args):
    device = 'cpu'
    with torch.no_grad():
        if args.dataset == "ogbn-mag":
            return load_mag(device, args)
        else:
            raise RuntimeError(f"Dataset {args.dataset} not supported")


def load_mag(device, args):
    from ogb.nodeproppred import DglNodePropPredDataset
    path = os.path.join(args.emb_path, f"TransE_mag")
    dataset = DglNodePropPredDataset(
        name="ogbn-mag", root=args.root)
    g, labels = dataset[0]
    splitted_idx = dataset.get_idx_split()
    train_nid = splitted_idx["train"]['paper']
    val_nid = splitted_idx["valid"]['paper']
    test_nid = splitted_idx["test"]['paper']
    features = g.nodes['paper'].data['feat']
    author_emb = torch.load(os.path.join(path, "author.pt"), map_location=torch.device("cpu")).float()
    topic_emb = torch.load(os.path.join(path, "field_of_study.pt"), map_location=torch.device("cpu")).float()
    institution_emb = torch.load(os.path.join(path, "institution.pt"), map_location=torch.device("cpu")).float()

    g = g.to(device)
    g.nodes["author"].data["feat"] = author_emb.to(device)
    g.nodes["institution"].data["feat"] = institution_emb.to(device)
    g.nodes["field_of_study"].data["feat"] = topic_emb.to(device)
    g.nodes["paper"].data["feat"] = features.to(device)
    paper_dim = g.nodes["paper"].data["feat"].shape[1]
    author_dim = g.nodes["author"].data["feat"].shape[1]
    if paper_dim != author_dim:
        paper_feat = g.nodes["paper"].data.pop("feat")
        rand_weight = torch.Tensor(paper_dim, author_dim).uniform_(-0.5, 0.5)
        g.nodes["paper"].data["feat"] = torch.matmul(paper_feat, rand_weight.to(device))
        print(f"Randomly project paper feature from dimension {paper_dim} to {author_dim}")

    labels = labels['paper'].to(device).squeeze()
    n_classes = int(labels.max() - labels.min()) + 1

    return g, labels, n_classes, train_nid, val_nid, test_nid



def preprocess_features(g, rel_subsets, args, device):
    # pre-process heterogeneous graph g to generate neighbor-averaged features
    # for each relation subsets
    num_paper, feat_size = g.nodes["paper"].data["feat"].shape
    new_feats = [torch.zeros(num_paper, len(rel_subsets), feat_size) for _ in range(args.num_hops + 1)]
    print("Start generating features for each sub-metagraph:")
    for subset_id, subset in enumerate(rel_subsets):
        print(subset)
        feats = gen_rel_subset_feature(g, subset, args, device)
        for i in range(args.num_hops + 1):
            feat = feats[i]
            new_feats[i][:feat.shape[0], subset_id, :] = feat
        feats = None
    return new_feats
