import os
import utils.function as uf
from models.GLEM.GLEM_utils import *
from utils.data import SeqGraph
from utils.modules import ModelConfig, SubConfig
from utils.settings import *
from importlib import import_module


class LMConfig(ModelConfig):
    def __init__(self, args=None):
        # ! INITIALIZE ARGS
        super(LMConfig, self).__init__('LMs')

        # ! LM Settings
        self.model = 'Bert'
        self.init_ckpt = 'Prt'
        self.init_ckpt = 'PrevEM'

        self.lr = 0.00002
        self.eq_batch_size = 36
        self.weight_decay = 0.01
        self.label_smoothing_factor = 0.1
        self.dropout = 0.1
        self.warmup_epochs = 0.2
        self.att_dropout = 0.1
        self.cla_dropout = 0.1
        self.cla_bias = 'T'
        self.grad_acc_steps = 2
        self.load_best_model_at_end = 'T'

        # ! glem Training settings
        self.is_augmented = False
        self.save_folder = ''
        self.emi_file = ''
        self.em_iter = 0
        self.ce_reduction = 'mean'

        self.feat_shrink = '100'
        self.feat_shrink = ''
        self.pl_weight = 0.5  # pseudo_label_weight
        self.pl_ratio = 0.5  # pseudo_label data ratio
        self.eval_patience = 100000
        self.is_inf = False
        self.md = None  # Tobe initialized in sub module
        # ! POST_INIT
        self.em_phase = 'LM'
        # to be called in the instances of LMConfig.

    # *  <<<<<<<<<<<<<<<<<<<< POST INIT FUNCS >>>>>>>>>>>>>>>>>>>>

    def _intermediate_args_init(self):
        """
        Parse intermediate settings that shan't be saved or printed.
        """
        self.mode = 'em' if self.emi_file != '' and self.em_iter >= 0 else 'pre_train'
        self.lm_meta_data_init()
        self.hf_model = self.md.hf_model
        self.hidden_dim = int(self.feat_shrink) if self.feat_shrink else self.md.hidden_dim
        self._lm = SubConfig(self, self.para_prefix)

        # * Init LM settings using pre-train folder
        self.lm = get_lm_info(self.save_folder, self.model)

        # * Overwrite LM settings if EM-iter information is provided
        if hasattr(self, 'emi_file') and self.emi_file:
            self.em_info = uf.pickle_load(self.emi_file)
            self.emi = EmIterInfo(self.em_info, self.em_iter)
            self.lm = self.emi.lm
            prt_emi = EmIterInfo(self.em_info, -1)
            self.prev_lm_ckpt = self.emi.lm.ckpt if self.emi.has_prev_iter else prt_emi.lm.ckpt
            self.prt_lm_ckpt = prt_emi.lm.ckpt
            self.pl_filter = self.emi.cf.pl_filter
            if self.is_augmented:
                # GLEM + Pre/Cir-train Inf
                # * Use GLEM settings
                if not self.is_inf:
                    self.pseudo_label_file = self.emi.iter_info.gnn_pred
            self.wandb_id = self.emi.cf.wandb_id

    def lm_meta_data_init(self):
        self.md = self.meta_data[self.model]

    def _exp_init(self):
        super()._exp_init()
        # ! Batch_size Setting
        max_bsz = self.md.max_bsz
        self.batch_size, self.grad_acc_steps = uf.calc_bsz_grad_acc(self.eq_batch_size, max_bsz.train, SV_INFO)
        self.inf_batch_size = uf.get_max_batch_size(SV_INFO.gpu_mem, max_bsz.inf)

    def _data_args_init(self):
        # Dataset
        self.lm_md = self.md
        self.data = SeqGraph(self)

    # *  <<<<<<<<<<<<<<<<<<<< SUB MODULES >>>>>>>>>>>>>>>>>>>>

    # *  <<<<<<<<<<<<<<<<<<<< PATH RELATED >>>>>>>>>>>>>>>>>>>>
    para_prefix = {
        'model': '', 'lr': 'lr', 'eq_batch_size': 'bsz',
        'weight_decay': 'wd', 'dropout': 'do', 'att_dropout': 'atdo', 'cla_dropout': 'cla_do', 'cla_bias': 'cla_bias',
        'epochs': 'e', 'warmup_epochs': 'we', 'eval_patience': 'ef',
        'load_best_model_at_end': 'load', 'init_ckpt': 'ckpt',
        'label_smoothing_factor': 'lsf', 'pl_weight': 'alpha', 'pl_ratio': '', 'ce_reduction': 'red', 'feat_shrink': ''}

    args_to_parse = list(para_prefix.keys())
    meta_data = None

    @property
    def parser(self):
        parser = super().parser
        parser.add_argument("-m", "--model", default='TinyBert')
        parser.add_argument("-A", "--is_augmented", action="store_true")
        parser.add_argument("-I", "--is_inf", action="store_true")
        return parser

    @property
    def out_dir(self):
        return f'{TEMP_PATH}{self.model}/ckpts/{self.dataset}/{self.model_cf_str}/'

    @property
    def model_cf_str(self):
        return self.emi.glem_prefix if self.mode == 'em' else self._lm.f_prefix


# ! LM Settings
LM_SETTINGS = {}
LM_MODEL_MAP = {
    'Deberta-large': 'Deberta',
    'TinyBert': 'Bert',
    'Roberta-large': 'RoBerta',
    'LinkBert-large': 'LinkBert',
    'Bert-large': 'Bert',
    'GPT2': 'GPT',
    'GPT2-large': 'GPT',
    'Electra-large': 'Electra',
    'Electra-base': 'Electra',
}


def get_lm_model():
    return LMConfig().parser.parse_known_args()[0].model


def get_lm_trainer(model):
    if model in ['GPT2','GPT2-large']:
        from models.LMs.GPT_trainer import GPTTrainer as LMTrainer
    else:
        from models.LMs.lm_trainer import LMTrainer as LMTrainer
    return LMTrainer


def get_lm_config(model):
    model = LM_MODEL_MAP[model] if model in LM_MODEL_MAP else model
    return import_module(f'models.LMs.{model}').Config


def get_lm_config_by_glem_args(glem_args):
    Config = get_lm_config(model := glem_args['lm_model'])
    parsed_args = glem_args_to_sub_module_args(glem_args, target_prefix='lm_')
    return Config(parsed_args)
