from importlib import import_module

import utils.function as uf
from models.GLEM.GLEM_utils import *
from utils.data import SeqGraph
from utils.modules import ModelConfig, SubConfig
from utils.settings import *
import numpy as np


class GNNConfig(ModelConfig):

    def __init__(self, args=None):
        super(GNNConfig, self).__init__('GNNs')

        # ! GNNs shared settings
        self.model = 'GCN'
        self.lr = 0.01
        self.dropout = 0.5
        self.n_layers = 2
        self.n_hidden = 256
        self.weight_decay = 5e-4
        self.norm = 'BN'

        # ! Additional GLEM settings
        self.input_norm = 'T'
        self.emi_file = ''
        self.em_iter = 0
        self.pl_weight = 0.5
        self.pl_ratio = 0.5  # pseudo_label data ratio
        self.ce_reduction = 'mean'
        self.label_input = 'T'

        # ! Post Init Settings
        self.em_phase = 'GNN'
        self.is_augmented = False
        # self._post_init(args)
        # to be called in the instances of GNNConfig.

    # *  <<<<<<<<<<<<<<<<<<<< POST INIT FUNCS >>>>>>>>>>>>>>>>>>>>
    def _intermediate_args_init(self):
        # ! EM Info
        if hasattr(self, 'emi_file') and self.emi_file:
            self.em_info = uf.pickle_load(self.emi_file)
            self.emi = EmIterInfo(self.em_info, self.em_iter)
            self.pl_filter = self.emi.cf.pl_filter
            self.feature_file = self.emi.iter_info.gnn_feature
            self.pseudo_label_file = self.emi.iter_info.lm_pred

            self.wandb_id = self.emi.cf.wandb_id
            self.wandb_prefix = f'EM-GNN/'
            self.gnn = self.emi.gnn
        else:
            # Load Pre-trained Huggingface Embedding
            self.is_augmented = False

    def _sub_conf_init(self):
        super()._sub_conf_init()
        self._gnn_sub_cf = SubConfig(self, self.para_prefix)

    def _exp_init(self):
        import torch as th
        super()._exp_init()
        self.device = th.device("cuda:0") if self.gpus != '-1' else th.device('cpu')

    # *  <<<<<<<<<<<<<<<<<<<< PATH RELATED >>>>>>>>>>>>>>>>>>>>
    para_prefix = {'model': '', 'n_layers': 'l', 'lr': 'lr', 'epochs': 'e', 'dropout': 'do', 'n_hidden': 'd', 'early_stop': 'es', 'weight_decay': 'wd', 'norm': 'norm', 'input_norm': 'in',
                   'label_input': 'li',
                   'pl_weight': 'alpha',
                   'pl_ratio': '',
                   'ce_reduction': 'red'}
    args_to_parse = list(para_prefix.keys())

    @property
    def parser(self):
        parser = super().parser
        parser.add_argument("-m", "--model", default='GCN')
        parser.add_argument("-A", "--is_augmented", action="store_true")
        return parser

    @property
    def model_cf_str(self):
        return self._gnn_sub_cf.f_prefix if self.emi is None else f'{self.emi.glem_prefix}'

    def _data_args_init(self):
        self.lm_md = self.em_info.lm_md
        self.feat_shrink = self.em_info.feat_shrink
        self.data = SeqGraph(self)

    @property
    def checkpoint_file(self):
        return f"{TEMP_PATH}{self.model}/{self.dataset}/{self.model_cf_str}.ckpt"

    # *  <<<<<<<<<<<<<<<<<<<< MISC >>>>>>>>>>>>>>>>>>>>


GNN_SETTINGS = {}


def get_gnn_model():
    return GNNConfig().parser.parse_known_args()[0].model


def get_gnn_trainer(model):
    if model in ['GCN', 'RevGAT']:
        from models.GNNs.gnn_trainer import GNNTrainer
    elif model in ['SAGE']:
        from models.GNNs.minibatch_trainer import BatchGNNTrainer as GNNTrainer
    elif model in ['SAGN']:
        from models.GNNs.SAGNTrainer import SAGN_Trainer as GNNTrainer
    elif model in ['GAMLP']:
        from models.GNNs.GAMLPTrainer import GAMLP_Trainer as GNNTrainer
    elif model in ['GAMLP_DDP']:
        from models.GNNs.GAMLP_DDP_Trainer import GAMLP_DDP_Trainer as GNNTrainer
    else:
        raise ValueError(f'GNN-Trainer for model {model} is not defined')
    return GNNTrainer


def get_gnn_config(model):
    return import_module(f'models.GNNs.{model}').Config


def get_gnn_config_by_glem_args(glem_args):
    Config = get_gnn_config(glem_args['gnn_model'])
    parsed_args = glem_args_to_sub_module_args(glem_args, target_prefix='gnn_')
    return Config(parsed_args)


# random partition graph
def random_partition_graph(num_nodes, cluster_number=10):
    import numpy as np
    parts = np.random.randint(cluster_number, size=num_nodes)
    return parts


def save_and_report_gnn_result(cf, pred, res):
    uf.save_memmap(pred.cpu().numpy(), uf.init_path(cf.emi.gnn.pred), np.float16)
    if cf.emi.iter < 0:
        # Save results for pre-training to disk be reported at main ct-loop
        uf.pickle_save(res, cf.gnn.result)
        cf.wandb_log({f'gnn_prt_{k}': v for k, v in res.items()})
    else:
        # Save results to wandb
        cf.wandb_log({**{f'GLEM/GNN_{k}': v for k, v in res.items()},
                      'EM-Iter': cf.emi.end})
        cf.em_info.gnn_res_list.append(res)
    cf.log(f'\nTrain seed{cf.seed} finished\nResults: {res}\n{cf}')
    uf.pickle_save(cf.em_info, cf.emi_file)


def log_graph_feature_source(cf):
    cf.log(f'Loaded feature from previous LM embedding {cf.feature_file}')
    if cf.is_augmented:
        cf.log(f'GNN trained by {cf.pseudo_label_file}')
        if cf.label_input == 'T':
            cf.log(f'Label concatenated as additional node features!')
    else:
        cf.log(f'GNN trained by ground truth labels')
