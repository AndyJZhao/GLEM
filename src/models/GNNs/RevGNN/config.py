from utils.data import SeqGraph
from utils.modules import ModelConfig
from utils.settings import *
from models.GNNs.gnn_utils import GNNConfig


class RevGNNConfig(GNNConfig):

    def __init__(self, args=None):
        super(RevGNNConfig, self).__init__(args)
        # ! RevGNN shared settings #default
        self.model = 'RevGNN'
        self.epochs = 1000
        self.lr = 0.003
        self.n_layers = 7
        self.n_hidden = 160
        self.dropout = 0.5
        self.weight_decay = 0.0
        self.norm = 'LN'
        self.early_stop = 300

        # ! Specific Settings
        self.num_groups = 2

        # ! Post Init Settings
        self._post_init(args)

    # *  <<<<<<<<<<<<<<<<<<<< PATH RELATED >>>>>>>>>>>>>>>>>>>>
    para_prefix = {**GNNConfig.para_prefix, 'num_groups': 'ngps'}
    args_to_parse = list(para_prefix.keys())

