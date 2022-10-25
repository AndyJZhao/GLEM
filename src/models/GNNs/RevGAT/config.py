from utils.data import SeqGraph
from utils.modules import ModelConfig
from utils.settings import *
from models.GNNs.gnn_utils import GNNConfig


class RevGATConfig(GNNConfig):

    def __init__(self, args=None):
        super(RevGATConfig, self).__init__(args)
        # ! RevGNN shared settings
        self.model = 'RevGAT'
        self.epochs = 2000
        self.lr = 0.002
        self.n_layers = 3
        self.n_hidden = 256
        self.dropout = 0.75
        self.weight_decay = 0.0

        # ! Specific Settings
        self.n_label_iters = 0
        self.mask_rate = 0.5
        self.n_heads = 3
        self.input_drop = 0.25
        self.attn_drop = 0.0
        self.edge_drop = 0.3
        self.alpha = 0.5
        self.temp = 1.0
        self.use_labels = False
        self.no_attn_dst = True
        self.use_norm = False
        self.group = 2
        self.log_every = 20

        self.save = 'exp'
        self.kd_dir = './kd'
        self.mode = 'teacher'
        # ! Out Exp
        self.save_pred = True

        # ! Post Init Settings
        self._post_init(args)

    # *  <<<<<<<<<<<<<<<<<<<< PATH RELATED >>>>>>>>>>>>>>>>>>>>
    para_prefix = {**GNNConfig.para_prefix, 'mask_rate': 'mr', 'n_heads': 'ns', 'alpha':'',
                   }
    args_to_parse = list(para_prefix.keys()) + ['mode', 'kd_dir',  'save', 'log_every'
                    , 'group', 'use_norm', 'no_attn_dst', 'use_labels', 'temp', 'edge_drop',
                    'attn_drop', 'input_drop', 'n_label_iters']

    #5, attn_drop=0.0, backbone='rev', cpu=False, dropout=0.75, edge_drop=0.3, gpu=0, group=2, input_drop=0.25, kd_dir='./kd', log_every=20, lr=0.002, mask_rate=0.5, mode='teacher', n_epochs=2000, n_heads=3, n_hidden=256, n_label_iters=1, n_layers=5, n_runs=10, no_attn_dst=True, plot_curves=False, save='log/kd-L5-DP0.75-H256-20210620-044728-d9034b17-88b2-45fb-bfd2-bdc7d6de3313', save_pred=False, seed=0, temp=1.0, use_labels=True, use_norm=True, wd=0)
#  Runned 10 times