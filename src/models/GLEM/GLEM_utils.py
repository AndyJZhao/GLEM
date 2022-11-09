from utils.settings import *
from utils.function.np_utils import save_memmap
from utils.function import init_path
import numpy as np
import math


class EmIterInfo:
    # ! 这里的逻辑是存放各个 EM-Iter 中不变的信息
    # ! GNN first/ LM first 的逻辑在别处存放

    def __init__(self, em_info, em_iter=-1):
        self.em_info, self.iter = em_info, em_iter
        self.__dict__.update(em_info.__dict__)
        self.cf = cf = SN(**em_info.cf)
        self.prt_lm = ilm = em_info.prt_lm
        self.temperature = cf.pseudo_temp

        # ! Inference info
        self.n_tr_nodes = n_tr_nodes = em_info.n_train_nodes
        self.n_pl_nodes = em_info.n_pl_nodes
        if cf.pl_filter:
            self.n_pl_nodes = int(float(cf.pl_filter) * self.n_pl_nodes)
        self.inf_tr_nodes = min(n_tr_nodes, cf.inf_tr_n_nodes)
        self.inf_stride = self.inf_tr_nodes / n_tr_nodes
        self.total_iters = int(math.ceil(cf.inf_n_epochs / self.inf_stride))
        if em_iter >= 0:
            self.start = em_iter * self.inf_stride
            self.end = min((em_iter + 1) * self.inf_stride, cf.inf_n_epochs)

        self.is_pretrain = em_iter < 0
        self.has_prev_iter = em_iter > 0

        # ! GNN Related
        if self.is_pretrain:  # Shared pretrain file in Mnt folder
            gnn_root = f'{MNT_TEMP_DIR}prt_gnn/{cf.dataset}/'
            if cf.gnn_ckpt:
                gnn_folder = f'{gnn_root}{self.cf.gnn_model}/{cf.gnn_ckpt}/'
            else:
                gnn_folder = f'{gnn_root}{self.cf.gnn_model}/{em_info.lm.model}/{em_info.gnn_cfg_str}/'
        else:
            gnn_root = f'{TEMP_PATH}glem_gnn/{cf.dataset}/'
            gnn_folder = f'{gnn_root}{em_info.glem_cfg_str}/'
        self.gnn = SN(folder=gnn_folder, pred=f'{gnn_folder}.pred',
                      result=f'{gnn_folder}.result')

        # ! LM Related
        if self.is_pretrain:  # Shared pretrain file in Mnt folder
            lm_root = f'{MNT_TEMP_DIR}prt_lm/{cf.dataset}/'
            lm_folder = f'{lm_root}{self.cf.lm_model}/{ilm.model}/'
        else:
            lm_root = f'{TEMP_PATH}glem_lm/{cf.dataset}/'
            lm_folder = f'{lm_root}{em_info.glem_cfg_str}/'
        self.lm = get_lm_info(lm_folder, self.cf.lm_model)

        # ! EM Info contains GNN feature, GNN target, LM target types
        # Returns tne model input and output types for E&M iters

        shared_em_settings = {
            'pre-train': ['Prt', '', ''],
            'iter>0': ['latest', 'latest', 'latest'],
        }

        em_settings = {
            'GNN-first': {  # LM has previous GNN
                **shared_em_settings,
                'iter-0': ['Prt', 'Prt', 'latest'],
            },
            'LM-first': {  # GNN has previous LM
                **shared_em_settings,
                'iter-0': ['latest', 'latest', 'Prt'],
            },
        }
        if self.iter < 0:
            self.em_iter_info_src = em_settings[self.cf.em_order]['pre-train']
        elif self.iter == 0:
            self.em_iter_info_src = em_settings[self.cf.em_order]['iter-0']
        else:  # self.iter > 0:
            self.em_iter_info_src = em_settings[self.cf.em_order]['iter>0']

    @property
    def iter_info(self):
        _ = self.em_iter_info_src
        prt_emi = EmIterInfo(self.em_info, -1)
        emi_map = lambda k: {'Prt': prt_emi, 'latest': self}[k]
        return SN(
            gnn_feature=emi_map(_[0]).lm.emb if _[0] else '',
            lm_pred=emi_map(_[1]).lm.pred if _[1] else '',
            gnn_pred=emi_map(_[2]).gnn.pred if _[2] else '')

    @property
    def inf_node_ranges(self):
        i, N, N_PL = self.iter, self.n_tr_nodes, self.n_pl_nodes
        # Here we get the
        get_range = lambda n, N: range(s - N if (s := i * n % (2 * N)) + n > 2 * N else s, e + N if (e := (i + 1) * n % (2 * N)) < n else e)
        n_train = self.inf_tr_nodes
        n_pl = math.ceil(n_train * self.em_info.lm.pl_ratio)
        tr_range = get_range(n_train, N) if n_train < N else range(N)
        pl_range = get_range(n_pl, N_PL) if n_pl < N_PL else range(N_PL)
        return tr_range, pl_range


def glem_args_to_sub_module_args(glem_args_dict, target_prefix='gnn_'):
    # Ignore other settings
    ignored = [_ for _ in ['gnn_', 'lm_'] if _ != target_prefix]
    # Rename target_prefix
    keep_arg = lambda k: max([int(_ in k) for _ in ignored]) == 0
    cf_rename = lambda k: k.split(target_prefix)[1] if target_prefix in k and target_prefix != '' else k

    sub_module_dict = {}
    for k, v in glem_args_dict.items():
        if keep_arg(k):
            if f'{target_prefix}{k}' in glem_args_dict:
                # Overwrite existing ct-configs if defined in submodule config
                # e.g. lm_model should overwrite the model attribute of ct
                print()
            else:
                key_name = cf_rename(k)
                sub_module_dict[key_name] = v
    return SN(**sub_module_dict)


def get_lm_info(lm_folder, model):
    return SN(folder=lm_folder,
              emb=f'{lm_folder}{model}.emb',
              pred=f'{lm_folder}{model}.pred',
              ckpt=f'{lm_folder}{model}.ckpt',
              result=f'{lm_folder}{model}.result')


def compute_loss(logits, labels, loss_func, is_gold=None, pl_weight=0.5, is_augmented=False):
    """
    Combine two types of losses: (1-α)*MLE (CE loss on gold) + α*Pl_loss (CE loss on pseudo labels)
    """
    import torch as th

    if is_augmented and ((n_pseudo := sum(~is_gold)) > 0):
        deal_nan = lambda x: 0 if th.isnan(x) else x
        mle_loss = deal_nan(loss_func(logits[is_gold], labels[is_gold]))
        pl_loss = deal_nan(loss_func(logits[~is_gold], labels[~is_gold]))
        loss = pl_weight * pl_loss + (1 - pl_weight) * mle_loss
    else:
        loss = loss_func(logits, labels)
    return loss
