import os
from random import randint

import torch.nn.functional as F
from transformers import AutoTokenizer

from utils.function import *
from utils.function.dgl_utils import *
from utils.settings import *
from utils.data.preprocess import tokenize_graph, load_graph_info
from copy import deepcopy
import numpy as np
from utils.data.preprocess import *


class SeqGraph():
    def __init__(self, cf):
        # Process split settings, e.g. -1/2 means first split
        self.cf = cf
        self.hf_model = cf.lm_md.hf_model
        self.device = None
        self.lm_emb_dim = self.cf.lm_md.hidden_dim if not self.cf.feat_shrink else int(self.cf.feat_shrink)
        self.name, process_mode = (_ := cf.dataset.split('_'))[0], _[1]  # e.g. name of "arxiv_TA" is "arxiv"
        process_mode = process_mode.strip('ind').strip('IND').strip('ogbInd')
        self.process_mode = process_mode
        self.md = info_dict = DATA_INFO[self.name]

        if cf.em_phase == 'GNN':
            self.label_as_feat = cf.is_augmented and cf.emi.iter >= 0 and cf.label_input == 'T'
            self.node_feat_dim = self.lm_emb_dim
            if self.label_as_feat:
                self.node_feat_dim += info_dict['n_labels']

        self.n_labels = info_dict['n_labels']
        self.__dict__.update(info_dict)
        self.subset_ratio = 1 if len(_) == 2 else float(_[2])
        self.cut_off = info_dict['cut_off'] if 'cut_off' in info_dict else 512

        self.label_keys = ['labels', 'is_gold']
        self.token_keys = ['attention_mask', 'input_ids', 'token_type_ids']
        self.ndata = {}

        # * GLEM-related files information
        self._g_info_folder = init_path(f"{DATA_PATH}{cf.dataset}/")
        self._g_info_file = f"{self._g_info_folder}graph.info "
        self._token_folder = init_path(f"{DATA_PATH}{self.name}{process_mode}_{self.hf_model}/")
        self._processed_flag = {
            'g_info': f'{self._g_info_folder}processed.flag',
            'token': f'{self._token_folder}processed.flag',
        }
        if hasattr(cf, 'ct'):  # Try to update f_info_file
            self._ct_folder = init_path(f"{cf.emi.glem_cfg_str}")
        self.g, self.split = None, None

        self.info = {
            'input_ids': SN(shape=(self.md['n_nodes'], self.md['max_length']), type=np.uint16),
            'attention_mask': SN(shape=(self.md['n_nodes'], self.md['max_length']), type=bool),
            'token_type_ids': SN(shape=(self.md['n_nodes'], self.md['max_length']), type=bool)
        }
        for k, info in self.info.items():
            info.path = f'{self._token_folder}{k}.npy'
        return

    def init(self):
        # ! Load sequence graph info which is shared by GNN and LMs
        cf = self.cf
        self.gi = g_info = load_graph_info(cf)
        self.__dict__.update(g_info.splits)
        self.n_nodes = g_info.n_nodes
        self.ndata.update({_: getattr(g_info, _) for _ in self.label_keys})
        if self.cf.dataset == 'paper_TA':
            pl_nodes = g_info.val_test
        else:
            pl_nodes = ~g_info.is_gold
        if 'ind' in cf.dataset or 'IND' in cf.dataset:
            # Remove test set inference in inductive setting
            pl_nodes[g_info.splits['test_x']] = False
            if cf.em_phase == 'GNN' and 'ind' in cf.dataset:
                self.test_x = self.valid_x
        self.pl_nodes = np.arange(self.n_nodes)[pl_nodes]  # val test
        self.labeled_nodes = np.arange(self.n_nodes)[g_info.is_gold]
        if self.cf.em_phase == 'LM':
            # LM, don't process graph
            tokenize_graph(self.cf)
            self._load_data_fields(self.token_keys)
        if hasattr(self.cf, 'pseudo_label_file') and self.cf.pseudo_label_file:  # Both GNN & LM
            self.info['pseudo_labels'] = SN(type=np.float16, path=self.cf.pseudo_label_file, shape=(self.n_nodes, self.n_labels))
            self._load_data_fields(['pseudo_labels'])
            if self.cf.pl_filter:
                temp_file = self.cf.model_cf_str + f'iter{self.cf.em_iter}.confident_nodes'
                if cf.local_rank <= 0:
                    # ! Load full-graph
                    print(f'Processing data on LOCAL_RANK #{cf.local_rank}...')
                    ul = self.pl_nodes
                    _ = th.tensor(self.ndata['pseudo_labels'][ul]).to(th.float32).softmax(1).max(1)[0].topk(int(float(self.cf.pl_filter) * len(ul))).indices
                    self.pl_nodes = ul[_]
                    pickle_save(self.pl_nodes, temp_file)
                else:
                    # If not main worker (i.e. Local_rank!=0), wait until data is processed and load
                    print(f'Waiting for tokenization on LOCAL_RANK #{cf.local_rank}')
                    while not os.path.exists(temp_file):
                        time.sleep(2)  # Check if processed every 2 seconds
                    print(f'Detected processed data, LOCAL_RANK #{cf.local_rank} start loading!')
                    time.sleep(5)  # Wait for file write for 5 seconds
                    self.pl_nodes = pickle_load(temp_file)
        self.device = cf.device  # if cf.local_rank<0 else th.device(cf.local_rank)

        return self

    def init_gnn_feature(self):
        if 'ogbInd' in self.cf.dataset:
            return
        if hasattr(self.cf, 'feature_file') and self.cf.feature_file:  # For GNN
            self.info['feature'] = SN(type=np.float16, path=self.cf.feature_file, shape=(self.n_nodes, self.lm_emb_dim))
            self._load_data_fields(['feature'])
            self.label_as_feat = self.cf.is_augmented and self.cf.emi.iter >= 0 and self.cf.label_input == 'T'

    def get_inf_aug_train_ids(self, gold_range, aug_range):
        permuted = [np.tile(np.random.permutation(x), 2) for x in [self.train_x, self.pl_nodes]]
        gold_nodes = permuted[0][gold_range]
        aug_nodes = permuted[1][aug_range]
        train_ids = np.random.permutation(np.concatenate([gold_nodes, aug_nodes]))
        assert len(set(gold_nodes)) == len(gold_nodes)
        assert len(set(aug_nodes)) == len(aug_nodes)
        self.cf.log(f'Trained on {gold_range} gold nodes and {aug_range} augmented nodes')
        return train_ids

    def get_sampled_aug_ids(self, num_nodes):
        # Return sampled pseudo_labels
        sampled = np.random.choice(self.pl_nodes, min(num_nodes, len(self.pl_nodes)), replace=False)
        return sampled

    def _load_data_fields(self, k_list):
        for k in k_list:
            i = self.info[k]
            try:
                self.ndata[k] = np.memmap(i.path, mode='r', dtype=i.type, shape=i.shape)
            except:
                raise ValueError(f'Shape not match {i.shape}')

    def save_g_info(self, g_info):
        pickle_save(g_info, self._g_info_file)
        pickle_save('processed', self._processed_flag['g_info'])
        return

    def is_processed(self, field):
        return os.path.exists(self._processed_flag[field])

    def _from_numpy(self, x, on_cpu=False):
        return th.from_numpy(np.array(x)) if on_cpu else th.from_numpy(np.array(x)).to(self.device)

    def _th_float(self, x, on_cpu=False):
        return self._from_numpy(x, on_cpu).to(th.float32)

    def is_gold(self, nodes, on_cpu=False):
        return self._from_numpy(self.ndata['is_gold'][nodes], on_cpu)

    def y_gold(self, nodes, on_cpu=False):
        labels = self._from_numpy(self.ndata['labels'][nodes], on_cpu).to(th.int64)
        return F.one_hot(labels, num_classes=self.n_labels).type(th.FloatTensor) if on_cpu \
            else F.one_hot(labels, num_classes=self.n_labels).type(th.FloatTensor).to(self.device)

    def y_hat(self, nodes, on_cpu=False):
        # Pseudo labels overwritten by gold
        is_gold = self.is_gold(nodes)
        y_pred = th.softmax(self._th_float(self.ndata['pseudo_labels'][nodes] / self.cf.emi.temperature, on_cpu), dim=-1)
        # Overwrite pseudo_y using gold_y
        y_pred[is_gold] = self.y_gold(th.tensor(nodes)[is_gold], on_cpu)
        return y_pred

    def node_feature(self, nodes, on_cpu=False):
        # Only called at GNN (M-) step
        if 'ogbInd' in self.cf.dataset:
            features = self.ogb_feat[nodes].cpu() if on_cpu else self.ogb_feat[nodes].to(self.device)
        else:
            features = self._th_float(self.ndata['feature'][nodes], on_cpu)
        if self.label_as_feat:
            # Concat feature and prediction
            features = th.cat((features, self.y_hat(nodes, on_cpu)), dim=1)
        return features

    def node_labels(self, nodes):
        return self.y_hat(nodes) if self.cf.is_augmented else self.y_gold(nodes)

    def __getitem__(self, k):
        return self.ndata[k]

    def get_tokens(self, node_id):
        # node_id = self.gi.IDs[node_id] if hasattr(self.gi, 'IDs') else node_id
        _load = lambda k: th.IntTensor(np.array(self.ndata[k][node_id]))
        # item = {k: _load(k) for k in self.token_keys if k != 'input_ids'}
        item = {}
        item['attention_mask'] = _load('attention_mask')
        item['input_ids'] = th.IntTensor(np.array(self['input_ids'][node_id]).astype(np.int32))
        if self.hf_model not in ['distilbert-base-uncased','roberta-base']:
            item['token_type_ids'] = _load('token_type_ids')
        return item


class SeqGraphDataset(th.utils.data.Dataset):  # Map style
    def __init__(self, data: SeqGraph, mode):
        super().__init__()
        self.d, self.mode = data, mode

    def __getitem__(self, node_id):
        if 'train' in self.mode:
            item = self.d.get_tokens(node_id)
        else:
            node_id = self.d.gi.IDs[node_id] if hasattr(self.d.gi, 'IDs') else node_id
            item = self.d.get_tokens(node_id)
        if 'train' in self.mode:
            item['is_gold'] = self.d.is_gold(node_id)
            if self.mode == 'train_augmented':
                # = Train on full-data with pseudo labels
                item['labels'] = self.d.y_hat(node_id)
            elif self.mode == 'train_gold':
                # = Train on gold-labels only
                item['labels'] = self.d.y_gold(node_id)
        if 'ids' in self.mode:
            node_id = np.where(self.d.gi.IDs == node_id)[0].item() if hasattr(self.d.gi, 'IDs') else node_id
            item['node_id'] = node_id
        return item

    def __len__(self):
        return self.d.n_nodes
