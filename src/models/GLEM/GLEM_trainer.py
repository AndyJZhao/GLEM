from os.path import exists as f_exists
import utils.function as uf
from datasets import load_metric

from models.LMs.lm_utils import *
from models.GNNs.gnn_utils import *
from models.GLEM.GLEM_utils import *
import torch as th

metric = load_metric('src/utils/function/hf_accuracy.py')


class GLEMTrainer():
    """Convert textural graph to text list"""

    def __init__(self, cf):
        from transformers import logging
        # logging.set_verbosity_warning()
        logging.set_verbosity_error()

        self.cf = cf
        self.logger = cf.logger
        self.log = cf.logger.log
        self.em_iter = -1  # For pretrain
        self.em_range = range(self.cf.emi.total_iters)
        if cf.em_range:
            start, end = [int(_) for _ in cf.em_range.split('-')]
            self.em_range = range(max(0, start), min(self.cf.emi.total_iters, end))
            self.log(f'External EM range given! Running {self.em_range}')

    @property
    def cur_emi(self):
        return EmIterInfo(self.cf.em_info, self.em_iter)

    def _em_iter_log(self, phase):
        self.log(f'\n <<<<<<<<<< EM-Iter-{self.em_iter + 1}/{self.cf.emi.total_iters} {phase} EM-Epoch:{self.cur_emi.start:.3f}-{self.cur_emi.end:.3f} >>>>>>>>>>')

    def _get_cmds(self, train_prefix, sub_cf, default_parser):
        # Combine model-sub-config with experimental config
        new_cf = SN(**{**self.cf.exp.model_conf.__dict__, **sub_cf.model_conf.__dict__}) if sub_cf is not None else SN()
        new_cf.emi_file = self.cf.emi_file
        new_cf.em_iter = self.em_iter
        cmd = f'{train_prefix} ' + uf.args_to_cmd(default_parser, new_cf, allow_unknown_args=True) + f' --seed={self.cf.seed}'
        return cmd

    def _pre_train_lm(self):
        # Pretrain language model
        prt_emi = EmIterInfo(self.cf.em_info, -1)
        if f_exists(ckpt := prt_emi.lm.ckpt):
            self.log(f'Previous pretrained-LM checkpoint exists: {ckpt}')
        else:
            self.log(f'\n <<<<<<<<<< LM-Pretraining >>>>>>>>>>')
            available_gpus = self.cf.gpus.split(',')
            gpus = ','.join(available_gpus[:min(self.cf.prt_lm.max_n_gpus, len(available_gpus))])
            cmd = f'{self.cf.lm_tr_prefix} -m{self.cf.lm_model} {self.cf.prt_lm.cmd} --save_folder={prt_emi.lm.folder} -d{self.cf.dataset} -g{gpus} {f"-wLM_Prt_{self.cf.dataset[:4]}" if self.cf.wandb_on else ""} --em_iter=-1'
            uf.run_command_parallel(cmd, gpus, self.log)
            th.cuda.empty_cache()

        # Inference pre-trained language model
        if f_exists(f := prt_emi.lm.emb) and f_exists(prt_emi.lm.pred) and f_exists(prt_emi.lm.result):
            self.log(f'Previous pretrained-LM emb and pred exists: {f}')
        else:
            self.log(f'\n <<<<<<<<<< LM-Pre-train Inference >>>>>>>>>>')
            self._inf_lm()
        prt_res = {f'GLEM/LM_{k}': v
                   for k, v in uf.pickle_load(prt_emi.lm.result).items()}
        self.cf.wandb_log({**prt_res, 'EM-Iter': 0}, log=True)

    def run_gnn_cmd(self, cmd):
        if self.cf.gnn.model in {'GAMLP_DDP'}:
            uf.run_command(cmd + ' -F', self.log)
            if 'ind' in self.cf.dataset:
                uf.run_command(cmd + f' -g{self.cf.gpus[0]}', self.log)
            else:
                uf.run_command_parallel(cmd, self.cf.gpus, self.log)
        else:
            uf.run_command(cmd, self.log)
        th.cuda.empty_cache()

    def _pre_train_gnn(self):
        # Pretrain language model
        prt_emi = EmIterInfo(self.cf.em_info, -1)
        if f_exists(pred := prt_emi.gnn.pred) and f_exists(prt_emi.gnn.result):
            self.log(f'Previous pretrained-GNN exists, pred: {pred}')
        else:
            self.log(f'\n <<<<<<<<<< GNN-Pretraining >>>>>>>>>>')
            cmd = self._get_cmds(self.cf.gnn_tr_prefix, self.cf.gnn, GNNConfig().parser)
            cmd.replace(f'--wandb_id={self.cf.wandb_id}', f' -wGNN_Prt_{self.cf.dataset[:4]}')
            self.run_gnn_cmd(cmd)

        prt_res = {f'GLEM/GNN_{k}': v
                   for k, v in uf.pickle_load(prt_emi.gnn.result).items()}
        self.cf.wandb_log({**prt_res, 'EM-Iter': 0}, log=True)

    def _inf_lm(self):
        cmd = self._get_cmds(self.cf.lm_inf_prefix, self.cf.lm, LMConfig().parser) + ' -I'
        # No need to train, therefore no parallel running needed

        uf.run_command_parallel(cmd, self.cf.gpus, self.log)
        th.cuda.empty_cache()

    def _inference(self):
        # ! LM training
        self._em_iter_log('LM Train')
        cmd = self._get_cmds(self.cf.lm_tr_prefix, self.cf.lm, LMConfig().parser) + ' -A'
        uf.run_command_parallel(cmd, self.cf.gpus, self.log)
        th.cuda.empty_cache()

        # ! LM Inference
        # Not last phase: Cir-train Inference
        self._em_iter_log('LM Inference')
        self._inf_lm()

    def _maximization(self):
        # ! GNN training
        self._em_iter_log('GNN')
        cmd = self._get_cmds(self.cf.gnn_tr_prefix, self.cf.gnn, GNNConfig().parser) + ' -A'
        self.run_gnn_cmd(cmd)

    def _final_report(self):
        def get_best_by_val_acc(res_list, prefix):
            results = max(res_list, key=lambda x: x['val_acc'])
            return {f'{prefix}_{k}': v for k, v in results.items()}

        self.logger.static_log(self.cf.model_conf)
        self.cf.wandb_init()
        em_info = uf.pickle_load(self.cf.emi_file)
        res_data = {**get_best_by_val_acc(em_info.gnn_res_list, 'gnn'), **get_best_by_val_acc(em_info.lm_res_list, 'lm')}
        self.logger.save(res_data)
        self.logger.log(f'GLEM-Training completed!\n{res_data}')
        # # ! Remove temp files
        # uf.silent_remove(self.cur_emi.gnn.folder)
        # uf.silent_remove(self.cur_emi.lm.folder)

    def glem_train(self):
        self._pre_train_lm()  # Get LM emb + pred
        self._pre_train_gnn()  # Get GNN pred (OGB)
        for self.em_iter in self.em_range:
            if self.cf.em_order == 'GNN-first':
                self._maximization()
                self._inference()
            else:  # LM-First
                self._inference()
                self._maximization()
            if self.em_iter == (self.em_range.stop - 1):
                self._final_report()
