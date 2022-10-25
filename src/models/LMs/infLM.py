import os.path as osp
import sys
from tqdm import tqdm

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))
from transformers import AutoModel, TrainingArguments, Trainer

from models.LMs.lm_utils import *
from models.LMs.model import *
from utils.data.datasets import SeqGraphDataset
from transformers import logging as trfm_logging
from ogb.nodeproppred import Evaluator

METRIC_LIST = ['accuracy']


class LmInfTrainer:
    """Convert textural graph to text list"""

    def __init__(self, cf):
        self.cf = cf
        # logging.set_verbosity_warning()
        # trfm_logging.set_verbosity_error()
        self.logger = cf.logger
        self.log = cf.logger.log
        self.d = SeqGraph(cf).init()
        self._evaluator = Evaluator(name=cf.data.ogb_name)
        self.evaluator = lambda preds, labels: self._evaluator.eval({
            "y_true": th.tensor(labels).view(-1, 1),
            "y_pred": th.tensor(preds).view(-1, 1),
        })["acc"]
        # ! memmap
        self.emb = np.memmap(uf.init_path(self.cf.emi.lm.emb), dtype=np.float16, mode='w+',
                             shape=(self.d.n_nodes, self.cf.hidden_dim))
        self.pred = np.memmap(uf.init_path(self.cf.emi.lm.pred), dtype=np.float16, mode='w+',
                              shape=(self.d.n_nodes, self.d.md['n_labels']))
        # ! Load BertConfigs Encoder + Decoder together and output emb + predictions
        bert_model = AutoModel.from_pretrained(self.cf.hf_model)
        self.model = BertClassifier(
            bert_model, self.cf.data.n_labels,
            loss_func=th.nn.CrossEntropyLoss(label_smoothing=self.cf.label_smoothing_factor, reduction=cf.ce_reduction), cla_bias=self.cf.cla_bias == 'T', pseudo_label_weight=self.cf.pl_weight,
            feat_shrink=self.cf.feat_shrink
        )
        # The reduction should be sum in case unbalanced gold and pseudo data
        self.model.load_state_dict(th.load(ckpt := self.cf.lm.ckpt, map_location='cpu'))
        self.log(f'Performing inference using LM model: {ckpt}')

    @th.no_grad()
    def inference_pred_and_emb(self):
        th.cuda.empty_cache()
        inference_dataset = SeqGraphDataset(self.d, mode='token_and_ids')
        # Save embedding and predictions

        # print prediiction维度
        inf_model = BertClaInfModel(self.model, self.emb, self.pred, feat_shrink=self.cf.feat_shrink)  # .to(self.cf.device)
        inf_model.eval()
        inference_args = TrainingArguments(
            output_dir=f'{self.cf.out_dir}inf/',
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.cf.inf_batch_size,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            local_rank=self.cf.local_rank,
            fp16_full_eval=True,
        )
        trainer = Trainer(model=inf_model, args=inference_args)
        trainer.predict(inference_dataset)
        uf.remove_file(f'{self.cf.out_dir}inf/')

        # Evaluate and save results
        eval = lambda x: self.evaluator(np.argmax(self.pred[x], -1), self.d['labels'][x])
        if hasattr(self.d.gi, 'IDs'):
            val_dict = dict((k, i) for i, k in enumerate(self.d.valid_x))
            self.d.valid_x = [i for i, x in enumerate(self.d.gi.IDs) if x in val_dict]
            test_dict = dict((k, i) for i, k in enumerate(self.d.test_x))
            self.d.test_x = [i for i, x in enumerate(self.d.gi.IDs) if x in test_dict]
        res = {'val_acc': eval(self.d.valid_x), 'test_acc': eval(self.d.test_x)}
        self.log(f'LM inference completed.\nfeature file: {self.cf.emi.lm.emb}\nPredictions file : {self.cf.emi.lm.pred}')
        if cf.emi.iter >= 0:
            # Save results to wandb
            cf.wandb_log({**{f'GraphVF/LM_{k}': v for k, v in res.items()},
                          'EM-Iter': cf.emi.end})
            cf.em_info.lm_res_list.append(res)
        else:  # Pretrain
            # Save results for pre-training to be reported at main ct-loop
            uf.pickle_save(res, cf.lm.result)
            cf.wandb_log({f'lm_prt_{k}': v for k, v in res.items()})
        cf.log(f'\nTrain seed{cf.seed} finished\nResults: {res}\n{cf}')
        uf.pickle_save(cf.em_info, cf.emi_file)


if __name__ == "__main__":
    # ! Init Arguments
    model = get_lm_model()
    Config = get_lm_config(model)
    args = Config().parse_args()
    cf = Config(args).init()

    # ! Load data and train
    trainer = LmInfTrainer(cf=cf)
    trainer.inference_pred_and_emb()
