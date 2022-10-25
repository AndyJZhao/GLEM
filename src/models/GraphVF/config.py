from models.GNNs.gnn_utils import *
from models.LMs.lm_utils import *
from models.GraphVF.gvf_utils import *
from utils.data import SeqGraph
from utils.modules.conf_utils import *


class GraphVFConfig(ModelConfig):

    def __init__(self, args=None):
        super(GraphVFConfig, self).__init__('GraphVF')
        # ! Circular Training settings
        self.inf_n_epochs = 1
        self.inf_tr_n_nodes = 200000
        self.em_n_iter_per_epoch = 1  # How many steps per iter
        self.em_order = 'LM-first'
        self.pseudo_temp = 1.0
        self.lm_model = 'TinyBert'
        self.gnn_model = 'GCN'
        self.pl_filter = '0.8'  # Top ratio
        self.pl_filter = ''
        self.em_range = '1-2'
        self.em_range = ''
        self.gnn_ckpt = 'ogb'
        self.gnn_ckpt = ''
        self.lm_ckpt = ''

        # ! POST_INIT
        self._post_init(args)

    # *  <<<<<<<<<<<<<<<<<<<< POST INIT FUNCS >>>>>>>>>>>>>>>>>>>>
    def _exp_init(self):
        import torch as th
        super()._exp_init()
        if self.gpus != '-1':  # GPU
            if self.local_rank >= 0:  # DDP
                th.cuda.set_device(self.local_rank)
                self.device = th.device(self.local_rank)
            else:  # Single GPU
                self.device = th.device("cuda:0")
        else:  # CPU
            self.device = th.device('cpu')
        lm_cf = get_lm_config_by_gvf_args(self.model_conf)
        gnn_cf = get_gnn_config_by_gvf_args(self.model_conf)

        self.lm = SubConfig(lm_cf, lm_cf.para_prefix)
        self.lm_md = lm_cf.md
        self.gnn = SubConfig(gnn_cf, gnn_cf.para_prefix)
        self.feat_shrink = self.lm.feat_shrink
        self.data = SeqGraph(self)
        from utils.data.preprocess import load_graph_info
        g_info = load_graph_info(self)
        self.prt_lm = lm_cf.md.prt_lm[self.data.name]
        n_pl_nodes = sum(~g_info.is_gold) if 'paper' not in self.dataset else len(g_info.splits['test_x']) + len(g_info.splits['valid_x'])
        if 'ind' in self.dataset or 'IND' in self.dataset:  # No test in ind settings
            n_pl_nodes -= len(g_info.splits['test_x'])
        # ! Save EM Info to file
        self.em_info = em_info = SN(
            gvf_prefix=self.f_prefix,
            gvf_cfg_str=self.model_cf_str,
            wandb_id=self.wandb_id,
            gnn_cfg_str=self.gnn.f_prefix,
            cf=self.model_conf,
            prt_lm=self.prt_lm,
            lm=self.lm.model_conf,
            lm_md=lm_cf.md,
            gnn=self.gnn.model_conf,
            gnn_res_list=[],
            lm_res_list=[],
            subset_ratio=self.data.subset_ratio,
            n_train_nodes=len(g_info.splits['train_x']),
            n_pl_nodes=n_pl_nodes,
            feat_shrink=self.feat_shrink
        )
        self.emi = EmIterInfo(em_info, 0)
        uf.pickle_save(em_info, self.emi_file)

    # *  <<<<<<<<<<<<<<<<<<<< SUB MODULES >>>>>>>>>>>>>>>>>>>>
    def _intermediate_args_init(self):
        """
        Parse intermediate settings that shan't be saved or printed.
        """
        self.inf_stride = 1 / self.em_n_iter_per_epoch
        self.em_phase = 'GVF-Main-Loop'
        SRC = 'src/models/'
        self.gnn_tr_prefix = f'{PYTHON} {SRC}GNNs/trainGNN.py'
        self.lm_tr_prefix = f'{PYTHON} {SRC}LMs/trainLM.py'
        self.lm_inf_prefix = f'{PYTHON} {SRC}LMs/infLM.py'

    def _sub_conf_init(self):
        super()._sub_conf_init()
        self.exp = SubConfig(self, ModelConfig.para_prefix)

    # *  <<<<<<<<<<<<<<<<<<<< PATH RELATED >>>>>>>>>>>>>>>>>>>>
    para_prefix = {'inf_n_epochs': 'MaxIter', 'inf_tr_n_nodes': 'inf_tr', 'pseudo_temp': 'temp', 'em_order': '', 'pl_filter': '', 'gnn_ckpt': '', 'lm_ckpt': '', }

    @property
    def emi_file(self):
        return self.res_file.replace('.json', '_em_info.pickle').replace(TEMP_RES_PATH, TEMP_PATH)

    @property
    def model_cf_str(self):
        return f'CR{self._model.f_prefix}_GNN{self.gnn.f_prefix}/LM{self.lm.f_prefix}'

    # *  <<<<<<<<<<<<<<<<<<<< MISC >>>>>>>>>>>>>>>>>>>>
    def parse_args(self):
        # ! Parse defined args
        parser = super().parser
        parser.add_argument("-m", "--lm_model", default='TinyBert')
        parser.add_argument("-n", "--gnn_model", default='GCN')
        defined_args = parser.parse_known_args()[0]
        defined_args.emi_file = ''

        # ! Reinitialize config by parsed experimental args
        lm_cf = get_lm_config_by_gvf_args(defined_args.__dict__)
        gnn_cf = get_gnn_config_by_gvf_args(defined_args.__dict__)
        add_undefined_args_to_parser(parser, self.model_conf, defined_args, '')
        for conf, prefix in [(lm_cf, 'lm_'), (gnn_cf, 'gnn_')]:
            add_undefined_args_to_parser(parser, conf.model_conf, defined_args, prefix, valid_args_list=conf.args_to_parse)
        return parser.parse_args()


def add_undefined_args_to_parser(parser, conf_dict, defined_args, arg_prefix='', valid_args_list=None):
    for arg, arg_val in conf_dict.items():
        if not hasattr(defined_args, arg_name := f'{arg_prefix}{arg}'):
            if valid_args_list is None or arg in valid_args_list:
                parser.add_argument(f"--{arg_name}", type=type(arg_val), default=arg_val)
    return parser
