import math

from datasets import load_metric
from transformers import AutoModel, EvalPrediction, TrainingArguments, Trainer
import utils.function as uf
from models.LMs.model import *
from models.GraphVF.gvf_utils import *
from utils.data.datasets import *
import torch as th

METRICS = {  # metric -> metric_path
    'accuracy': 'src/utils/function/hf_accuracy.py'
}


class LMTrainer():
    """Convert textural graph to text list"""

    def __init__(self, cf):
        self.cf = cf
        # logging.set_verbosity_warning()
        from transformers import logging as trfm_logging
        trfm_logging.set_verbosity_error()
        self.logger = cf.logger
        self.log = cf.logger.log
        self.update_ratio = 1

    @uf.time_logger
    def train(self):
        # ! Prepare data
        self.d = d = SeqGraph(cf := self.cf).init()
        gold_data = SeqGraphDataset(self.d, mode='train_gold')
        subset_data = lambda sub_idx: th.utils.data.Subset(gold_data, sub_idx)
        self.datasets = {_: subset_data(getattr(d, f'{_}_x'))
                         for _ in ['train', 'valid', 'test']}
        self.metrics = {m: load_metric(m_path) for m, m_path in METRICS.items()}

        if cf.is_augmented:
            # Augment Label if Cir-train
            warmup_steps = 0  # No warmup (already warmed up at pre-training step)
            # Sample visible data for current EM-Iter
            init_random_state(cf.seed)
            train_ids = d.get_inf_aug_train_ids(*cf.emi.inf_node_ranges)
            _ = SeqGraphDataset(d, mode='train_augmented')
            self.train_data = th.utils.data.Subset(_, train_ids)
            max_pl_ratio = len(d.pl_nodes) / len(d.labeled_nodes)
            pl_ratio = min(cf.pl_ratio, max_pl_ratio)
            eval_steps = (1 + pl_ratio) * cf.eval_patience // cf.eq_batch_size
        else:
            # Pretrain on gold data
            self.train_data = self.datasets['train']
            train_steps = len(d.train_x) // cf.eq_batch_size + 1
            warmup_steps = int(cf.warmup_epochs * train_steps)
            eval_steps = cf.eval_patience // cf.eq_batch_size

        # ! Load bert and build classifier
        bert_model = AutoModel.from_pretrained(cf.hf_model)
        self.model = BertClassifier(
            bert_model, cf.data.n_labels,
            pseudo_label_weight=cf.pl_weight if cf.is_augmented else 0,
            dropout=cf.cla_dropout,
            loss_func=th.nn.CrossEntropyLoss(label_smoothing=cf.label_smoothing_factor, reduction=cf.ce_reduction),
            cla_bias=cf.cla_bias == 'T',
            is_augmented=cf.is_augmented,
            feat_shrink=cf.feat_shrink
        )
        if cf.local_rank <= 0:
            trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
            print(f"!!!!!!!!!!!!!!!!! LM Model parameters are {trainable_params}")
        if cf.is_augmented:
            if cf.init_ckpt == 'PrevEM':
                # ! Load previous LM model
                self.model.load_state_dict(temp := th.load(cf.prev_lm_ckpt, map_location='cpu'))
                del temp
            elif cf.init_ckpt == 'Prt':
                self.model.load_state_dict(temp := th.load(cf.prt_lm_ckpt, map_location='cpu'))
                del temp
            elif cf.init_ckpt == 'None':
                pass
            elif cf.init_ckpt == 'EM':
                # ! Don't Load previous Prt LM model
                if cf.emi.iter > 0:
                    print(f'-----------cf.emi.iter = {cf.emi.iter}, load from Prev')
                    self.model.load_state_dict(temp := th.load(cf.prev_lm_ckpt, map_location='cpu'))
                    del temp
                else:
                    print(f'-----------cf.emi.iter = {cf.emi.iter}, load from None')
                    pass
            else:
                raise NotImplementedError(cf.init_ckpt)
            load_best_model_at_end = cf.load_best_model_at_end == 'T'
        else:
            load_best_model_at_end = True
        self.model.config.hidden_dropout_prob = cf.dropout
        self.model.config.attention_dropout_prob = cf.att_dropout

        training_args = TrainingArguments(
            output_dir=cf.out_dir,
            evaluation_strategy='steps',
            eval_steps=eval_steps,
            save_strategy='steps',
            save_steps=eval_steps,
            learning_rate=cf.lr, weight_decay=cf.weight_decay,
            load_best_model_at_end=load_best_model_at_end, gradient_accumulation_steps=cf.grad_acc_steps,
            save_total_limit=1,
            report_to='wandb' if cf.wandb_on else None,
            per_device_train_batch_size=cf.batch_size,
            per_device_eval_batch_size=cf.batch_size * 10,
            warmup_steps=warmup_steps,
            disable_tqdm=False,
            dataloader_drop_last=True,
            num_train_epochs=cf.epochs,
            local_rank=cf.local_rank,
            dataloader_num_workers=0,
            fp16=True,  # if cf.hf_model=='microsoft/deberta-large' else False
        )

        # ! Get dataloader

        def compute_metrics(pred: EvalPrediction):
            predictions, references = pred.predictions.argmax(1), pred.label_ids.argmax(1)
            return {m_name: metric.compute(predictions=predictions, references=references) for m_name, metric in self.metrics.items()}

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_data,
            eval_dataset=self.datasets['valid'],
            compute_metrics=compute_metrics,
        )
        self.eval_phase = 'Eval'
        self.trainer.train()
        # ! Save bert
        # self.model.save_pretrained(cf.out_ckpt, self.model.state_dict())
        # ! Save BertClassifer Save model parameters
        if cf.local_rank <= 0:
            th.save(self.model.state_dict(), uf.init_path(cf.lm.ckpt))
        # uf.remove_file(f'{cf.out_dir}')
        self.log(f'LM saved to {cf.lm.ckpt}')

    def eval_and_save(self):
        def get_metric(split):
            self.eval_phase = 'Test' if split == 'test' else 'Eval'
            mtc_dict = self.trainer.predict(self.datasets[split]).metrics
            ret = {f'{split}_{_}': mtc_dict[m][_] for m in mtc_dict if (_ := m.split('_')[-1]) in METRICS}
            return ret

        cf = self.cf
        res = {**get_metric('valid'), **get_metric('test')}
        res = {'val_acc': res['valid_accuracy'], 'test_acc': res['test_accuracy']}
        if cf.is_augmented:
            cf.wandb_log({**{f'GraphVF/LM_{k}': v for k, v in res.items()},
                          'EM-Iter': cf.emi.end})
            cf.em_info.lm_res_list.append(res)
            uf.pickle_save(cf.em_info, cf.emi_file)
        else:  # Pretrain
            # Save results for pre-training to be reported at main ct-loop
            uf.pickle_save(res, cf.lm.result)
            cf.wandb_log({f'lm_prt_{k}': v for k, v in res.items()})

        self.log(f'\nTrain seed{cf.seed} finished\nResults: {res}\n{cf}')
