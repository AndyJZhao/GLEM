import os

from utils.data import SeqGraph
from utils.modules import ModelConfig
from utils.settings import *
from models.GNNs.gnn_utils import GNNConfig


class GAMLPConfig(GNNConfig):

    def __init__(self, args=None):
        super(GAMLPConfig, self).__init__(args)
        # E.g. different default lr, or fanouts
        self.model = 'GAMLP'
        self.n_hidden = 1024  # ok
        self.lr = 0.001
        self.dropout = 0.5  # ok
        self.input_dropout = 0.2  ##ok
        self.att_dropout = 0.5  # ok
        self.label_drop = 0.
        self.num_hops = 5  # ok
        self.alpha = 0.5  # ok
        self.n_layers = 4  # n_layers_2 ok
        self.n_layers_3 = 4
        self.act = 'leaky_relu'  # ok
        self.pre_process = True  # OK
        self.residual = True  # ok
        self.pre_dropout = False  # ok
        self.bns = True  # ok
        self.average = 'T0'
        self.prt_batch_size = 50000 # ok
        self.aug_batch_size = 50000
        self.eval_batch_size = 200000
        self.epochs = 400  # ok
        self.weight_decay = 0.0
        self.early_stop = 300
        self.label_num_hops = 9


        # ! Post Init Settings
        self._post_init(args)

    # *  <<<<<<<<<<<<<<<<<<<< PATH RELATED >>>>>>>>>>>>>>>>>>>>
    para_prefix = {**GNNConfig.para_prefix, 'input_dropout': 'ind', 'att_dropout': 'atd', 'num_hops': 'nps', 'alpha': 'a', 'act': '', 'prt_batch_size': 'bz', 'aug_batch_size': 'az',
                   'n_layers_3': 'l3', 'label_drop': 'ld', 'label_num_hops': '', 'average': ''}
    args_to_parse = list(para_prefix.keys())

    @property
    def processed_file(self):
        return f'{self.model_cf_str}iter{self.em_iter}.processed'

    def processed_feat(self, hop):
        return f'{self.model_cf_str}neighbor_averaged{hop}.feature'

    @property
    def is_processed(self):
        return os.path.exists(self.processed_file)
