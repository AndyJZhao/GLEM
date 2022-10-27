import gc
import os
import time

import dgl
import numpy as np
import pandas as pd
import torch as th
from transformers import AutoTokenizer
import time
import utils.function as uf
from utils.function.dgl_utils import sample_nodes
from utils.settings import *
from tqdm import tqdm
from copy import deepcopy
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, dropout_adj


def _subset_graph(g, cf, sup):
    splits = ['train', 'valid', 'test']
    if cf.data.subset_ratio != 1:
        subset = lambda x: x[:round(len(x) * cf.data.subset_ratio)].tolist()
        split_ids = {_ + '_x': subset(sup[_ + '_x']) for _ in splits}
        seed_nodes = th.tensor(sum([split_ids[_ + '_x'] for _ in splits], []))
        node_subset = sample_nodes(g, seed_nodes, [-1])[0]
        # g.ndata['labels'] = th.from_numpy(sup['labels'])
        g = dgl.node_subgraph(g, node_subset)
        # new_split_ids = {_: [] for _ in splits}
        # for i in range(g.num_nodes()):
        #     get_split=lambda _: g.ndata['_ID'][i] in split_ids[_ + '_x']
        #     for split in splits:
        #         if get_split(split):
        #             new_split_ids[split].append(i)
        # split_ids = {f'{_}_x': new_split_ids[_] for _ in splits}
    else:
        split_ids = {f'{_}_x': sup[_ + '_x'] for _ in splits}
    split_len = {_: len(split_ids[f'{_}_x']) for _ in splits}
    log_func = cf.log if hasattr(cf, 'log') else print
    log_func(f'Loaded dataset {cf.dataset} with {split_len} and {g.num_edges()} edges')
    return g, split_ids


def plot_length_distribution(node_text, tokenizer, g):
    sampled_ids = np.random.permutation(g.nodes())[:10000]
    get_text = lambda n: node_text.iloc[n]['text'].tolist()
    tokenized = tokenizer(get_text(sampled_ids), padding='do_not_pad').data['input_ids']
    node_text['text_length'] = node_text.apply(lambda x: len(x['text'].split(' ')), axis=1)
    pd.Series([len(_) for _ in tokenized]).hist(bins=20)
    import matplotlib.pyplot as plt
    plt.show()


def tokenize_graph(cf):
    # = Tokenization on Full Graph
    full_dict = deepcopy(cf.model_conf)
    full_dict['dataset'] = '_'.join(full_dict['dataset'].split('_')[:2])
    full_cf = cf.__class__(SN(**full_dict)).init()
    d = full_cf.data
    if not d.is_processed('token'):
        if cf.local_rank <= 0:
            # ! Load full-graph
            print(f'Processing data on LOCAL_RANK #{cf.local_rank}...')
            g_info = load_graph_info(full_cf)
            print(f'Loaded graph structure, start tokenization...')
            if d.ogb_name == 'ogbn-products':
                from utils.data.preprocess_product import _tokenize_ogb_product
                _tokenize_ogb_product(d, g_info.labels)
            elif d.ogb_name == 'ogbn-papers100M':
                from utils.data.preprocess_paper import _tokenize_ogb_paper_datasets
                _tokenize_ogb_paper_datasets(d, g_info.labels)
            else:
                _tokenize_ogb_arxiv_datasets(d, g_info.labels)
            print(f'Tokenization finished on LOCAL_RANK #{cf.local_rank}')
        else:
            # If not main worker (i.e. Local_rank!=0), wait until data is processed and load
            print(f'Waiting for tokenization on LOCAL_RANK #{cf.local_rank}')
            while not d.is_processed('token'):
                time.sleep(2)  # Check if processed every 2 seconds
            print(f'Detected processed data, LOCAL_RANK #{cf.local_rank} start loading!')
            time.sleep(5)  # Wait for file write for 5 seconds
    else:
        cf.log(f'Found processed {cf.dataset}.')


def process_graph_structure(g, cf):

    a = time.time()
    g = dgl.to_bidirected(g)
    g_info = cf.data.gi
    if cf.data.subset_ratio < 1:
        g = dgl.node_subgraph(g, cf.data.gi.IDs)
        g = g.remove_self_loop().add_self_loop()
    if cf.model in {'RevGAT','GCN'}:
        # add self-loop
        print(f"Using GAT based methods,total edges before adding self-loop {g.number_of_edges()}")
        g = g.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {g.number_of_edges()}")
    print(f'process_graph OK!, spend {time.time() - a}')
    if 'ind' in cf.dataset:
        visible_nodes = list(set(g.nodes().numpy().tolist()) - set(g_info.splits['test_x']))
        g = dgl.node_subgraph(g, visible_nodes)
    if 'IND' in cf.dataset or 'Ind' in cf.dataset:
        test_ids = g_info.splits['test_x']
        edges_to_rm = th.cat((g.in_edges(test_ids, form='eid'), g.out_edges(test_ids, form='eid')))
        g = dgl.remove_edges(g, edges_to_rm)
        g = g.remove_self_loop().add_self_loop()
    return g  # .to(cf.device)


def process_pyg_graph_structure(data, cf):
    path = '../adj_gcn.pt'
    a = time.time()
    if osp.exists(path):
        adj = th.load(path)
    else:
        N = data.num_nodes
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        row, col = data.edge_index
        print('Computing adj...')
        adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
        adj = adj.set_diag()
        deg = adj.sum(dim=1).to(th.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        th.save(adj, path)

    adj = adj.to_scipy(layout='csr')
    print(f'process_graph OK!, spend {time.time() - a}')
    del data
    return adj


def load_ogb_graph_structure_only(cf):
    from ogb.nodeproppred import DglNodePropPredDataset
    data = DglNodePropPredDataset(cf.data.ogb_name, root=uf.init_path(cf.data.raw_data_path))
    g, labels = data[0]
    split_idx = data.get_idx_split()
    labels = labels.squeeze().numpy()
    return g, labels, split_idx


def load_pyg_graph_structure_only(cf):
    from ogb.nodeproppred import PygNodePropPredDataset
    dataset = PygNodePropPredDataset(cf.data.ogb_name, root=uf.init_path(cf.data.raw_data_path))
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    return data, split_idx


def load_graph_info(cf):
    d = cf.data
    # ! Process Full Graph
    if not d.is_processed('g_info'):
        # Load OGB
        if cf.local_rank <= 0:
            g, labels, split_idx = load_ogb_graph_structure_only(cf)
            # Process and save supervision
            splits = {**{f'{_}_x': split_idx[_].numpy() for _ in ['train', 'valid', 'test']}, 'labels': labels}
            is_gold = np.zeros((g.num_nodes()), dtype=bool)
            val_test = np.zeros((g.num_nodes()), dtype=bool)
            g, splits = _subset_graph(g, cf, splits)
            is_gold[splits['train_x']] = True
            val_test[splits['valid_x']] = True
            val_test[splits['test_x']] = True
            g_info = SN(splits=splits, labels=labels, is_gold=is_gold, n_nodes=g.num_nodes(), val_test=val_test)
            if d.subset_ratio < 1:
                g_info.IDs = g.ndata['_ID'].numpy()
                g_info.labels = g_info.labels[g_info.IDs]
                g_info.is_gold = g_info.is_gold[g_info.IDs]
                g_info.val_test = g_info.val_test[g_info.IDs]
            d.save_g_info(g_info)
            del g
        else:
            # If not main worker (i.e. Local_rank!=0), wait until data is processed and load
            print(f'Waiting for feature processing on LOCAL_RANK #{cf.local_rank}')
            while not d.is_processed('g_info'):
                time.sleep(2)  # Check if processed every 2 seconds
            print(f'Detected processed feature, LOCAL_RANK #{cf.local_rank} start loading!')
            time.sleep(5)  # Wait f
    g_info = uf.pickle_load(d._g_info_file)
    return g_info


def _tokenize_ogb_arxiv_datasets(d, labels, chunk_size=50000):
    def merge_by_ids(meta_data, node_ids, categories):
        meta_data.columns = ["ID", "Title", "Abstract"]
        # meta_data.drop([0, meta_data.shape[0] - 1], axis=0, inplace=True)  # Drop first and last in Arxiv full dataset processing
        meta_data["ID"] = meta_data["ID"].astype(np.int64)
        meta_data.columns = ["mag_id", "title", "abstract"]
        data = pd.merge(node_ids, meta_data, how="left", on="mag_id")
        data = pd.merge(data, categories, how="left", on="label_id")
        return data

    def read_ids_and_labels(data_root):
        category_path_csv = f"{data_root}/mapping/labelidx2arxivcategeory.csv.gz"
        paper_id_path_csv = f"{data_root}/mapping/nodeidx2paperid.csv.gz"  #
        paper_ids = pd.read_csv(paper_id_path_csv)
        categories = pd.read_csv(category_path_csv)
        categories.columns = ["ID", "category"]  # 指定ID 和 category列写进去
        paper_ids.columns = ["ID", "mag_id"]
        categories.columns = ["label_id", "category"]
        paper_ids["label_id"] = labels
        return categories, paper_ids  # 返回类别和论文ID

    def process_raw_text_df(meta_data, node_ids, categories):

        data = merge_by_ids(meta_data.dropna(), node_ids, categories)
        data = data[~data['title'].isnull()]
        text_func = {
            'TA': lambda x: f"Title: {x['title']}. Abstract: {x['abstract']}",
            'T': lambda x: x['title'],
        }
        # Merge title and abstract
        data['text'] = data.apply(text_func[d.process_mode], axis=1)
        data['text'] = data.apply(lambda x: ' '.join(x['text'].split(' ')[:d.cut_off]), axis=1)
        return data['text']

    from ogb.utils.url import download_url, extract_zip
    # Get Raw text path
    assert d.ogb_name in ['ogbn-arxiv', 'ogbn-papers100M']
    print(f'Loading raw text for {d.ogb_name}')
    raw_text_path = download_url(d.raw_text_url, d.data_root)

    tokenizer = AutoTokenizer.from_pretrained(d.hf_model)
    token_info = {k: np.memmap(d.info[k].path, dtype=d.info[k].type, mode='w+', shape=d.info[k].shape)
                  for k in d.token_keys}
    categories, node_ids = read_ids_and_labels(d.data_root)
    for meta_data in tqdm(pd.read_table(raw_text_path, header=None, chunksize=chunk_size, skiprows=[0])):
        text = process_raw_text_df(meta_data, node_ids, categories)
        print(text.index, text)
        tokenized = tokenizer(text.tolist(), padding='max_length', truncation=True, max_length=512).data
        for k in d.token_keys:
            token_info[k][text.index] = np.array(tokenized[k], dtype=d.info[k].type)
    uf.pickle_save('processed', d._processed_flag['token'])
    return
