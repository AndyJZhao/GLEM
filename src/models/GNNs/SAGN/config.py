import os

from utils.data import SeqGraph
from utils.modules import ModelConfig
from utils.settings import *
from models.GNNs.gnn_utils import GNNConfig


class SAGN_Config(GNNConfig):

    def __init__(self, args=None):
        super(SAGN_Config, self).__init__(args)
        # E.g. different default lr, or fanouts
        self.model = 'SAGN'
        self.n_hidden = 256 #ok
        self.num_hops = 3 #ok
        self.num_heads = 1 #ok
        self.lr = 0.001 #ok
        self.weight_style = 'attention' #default
        self.alpha = 0.5 #default
        self.focal = 'first' #default
        self.dropout = 0.5 #ok
        self.input_drop = 0.2 #ok
        self.att_drop = 0.4  #ok
        self.zero_inits = True #ok
        self.position_emb = False #default
        self.mlp_layer = 1 #ok
        self.label_mlp_layer = 4
        self.label_drop = 0.5 #ok
        self.label_residual = False # default
        self.residual = True #ok
        self.epochs = 1000 #ok
        self.weight_decay = 0.0 #default
        self.early_stop = 300 #ok
        self.label_num_hops = 9 #ok
        self.batch_size=50000 #ok
        self.eval_batch_size=100000 #无关
        self.pre_process = True #ok
        #! Add
        self.tem = 0.5 #ok
        self.lam = 0.5 #ok
        self.decay = 0.9 #default

        #Mean_teacher
        self.ema=True #ok
        self.mean_teacher=True #ok
        self.ema_decay=0.99 #ok
        self.adap=True #ok
        self.sup_lam=1.0
        self.kl=True #ok
        self.kl_lam=0.02 #ok
        self.top=0.85 #ok
        self.down=0.75 #ok
        self.warm_up=100 #ok
        self.gap=20 #ok

        # ! Post Init Settings
        self._post_init(args)

    # *  <<<<<<<<<<<<<<<<<<<< PATH RELATED >>>>>>>>>>>>>>>>>>>>
    para_prefix = {**GNNConfig.para_prefix, 'batch_size': 'bsz', 'warm_up': 'wp', 'top': '', 'down': '','kl_lam': "",
                   'ema_decay': ''}
    args_to_parse = list(para_prefix.keys())

    @property
    def processed_file(self):
        return f'{self.model_cf_str}iter{self.em_iter}.processed'

    def processed_feat(self, hop):
        return f'{self.model_cf_str}neighbor_averaged{hop}.feature'

    @property
    def is_processed(self):
        return os.path.exists(self.processed_file)
