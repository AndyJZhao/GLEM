from utils.data import SeqGraph
from utils.modules import ModelConfig
from utils.settings import *
from models.GNNs.gnn_utils import GNNConfig


class EnGCNConfig(GNNConfig):

    def __init__(self, args=None):
        super(EnGCNConfig, self).__init__(args)
        # This will be completed in future
        self.model = 'EnGCN'
        self.n_layers = 8 #SAGN_layers
        self.num_mlp_layers = 3
        self.dropout = 0.2
        self.lr = 0.01
        self.n_hidden = 512
        self.weight_decay = 0.0
        self.batch_size = 10000
        self.SLE_threshold = 0.8
        self.tosparse = True

        # ! Post Init Settings
        self._post_init(args)

    # *  <<<<<<<<<<<<<<<<<<<< PATH RELATED >>>>>>>>>>>>>>>>>>>>
    para_prefix = {**GNNConfig.para_prefix, 'batch_size': 'bsz', 'SLE_threshold': 'sle',
                   'num_mlp_layers': 'ml'}
    args_to_parse = list(para_prefix.keys())