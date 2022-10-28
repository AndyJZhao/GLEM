from ogb.nodeproppred import Evaluator

import utils.function as uf
from models.GNNs.gnn_utils import *
from models.GNNs.EnGCN.model import EnGCN
from models.GNNs.EnGCN.config import EnGCNConfig
from models.GraphVF.gvf_utils import *
from models.GNNs.gnn_utils import *
from utils.data.datasets import *
from utils.data.preprocess import *
from utils.modules.early_stopper import EarlyStopping
from time import time

LOG_FREQ = 10


class EnGCNTrainer():
    def __init__(self, cf: EnGCNConfig):
        self.cf, self.logger = cf, cf.logger
        self.log = self.cf.logger.log
        self.wandb_prefix = cf.wandb_prefix if hasattr(cf, 'wandb_prefix') else ''
        self.is_ind = 'ind' in cf.dataset

        # ! Load data
        self.d = d = self.cf.data.init()
        d.init_gnn_feature()
        self.g = load_ogb_graph_structure_only(cf)[0]
        if 'ogbInd' in self.cf.dataset:
            self.d.ogb_feat = self.g.ndata['feat']
        self.g = process_graph_structure(self.g, cf).to(cf.device)
        self.train_x, self.val_x, self.test_x = [
            th.tensor(getattr(d, f'{_}_x')).to(cf.device) for _ in ['train', 'valid', 'test']]
        if self.is_ind:
            self.features = self.d.node_feature(self.g.ndata['_ID'].cpu().numpy()).to(cf.device)
        else:
            self.features = self.d.node_feature(range(self.d.n_nodes)).to(cf.device)
        self.gold_labels = th.from_numpy(self.d['labels']).to(th.int64).to(cf.device)
        if self.cf.is_augmented:
            self.pseudo_labels = self.d.y_hat(range(self.d.n_nodes))
        self.is_gold = self.d.is_gold(range(self.d.n_nodes))
        log_graph_feature_source(self.cf)

        # ! Trainer init
        if cf.model == 'EnGCN':
            self.model = EnGCN(self.features.shape[1], cf.n_hidden, cf.data.n_labels, cf.num_mlp_layers, cf.dropout).to(cf.device)
            self.optimizer = th.optim.Adam(self.model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
        else:
            ValueError(f'Unimplemented GNNs model {cf.model}!')
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f'!!!!!GNN Phase, trainable_params are {trainable_params}')
        self.stopper = EarlyStopping(patience=cf.early_stop, path=cf.checkpoint_file) if cf.early_stop > 0 else None
        self.loss_func = th.nn.CrossEntropyLoss(reduction=cf.ce_reduction)
        self._evaluator = Evaluator(name=cf.data.ogb_name)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels.view(-1, 1)}
        )["acc"]

    def _forward(self, *args):
        logits = self.model(*args)  # small-graph
        if self.is_ind:
            # Inductive: map subseted graph to full graph
            global_logits = th.zeros(self.d.n_nodes, logits.shape[1]).to(self.cf.device)
            global_logits[self.g.ndata['_ID']] = logits
            return global_logits
        else:
            return logits

    def _train(self):
        # ! Shared
        self.model.train()
        self.optimizer.zero_grad()
        # ! Specific
        logits = self._forward(self.g, self.features)
        if self.cf.is_augmented and self.cf.pl_ratio > 0:
            sampled = self.d.get_sampled_aug_ids(int(len(self.d.train_x) * self.cf.pl_ratio))
            sampled = np.concatenate((self.train_x.cpu().numpy(), sampled))
            loss = compute_loss(logits[sampled], self.pseudo_labels[sampled], self.loss_func, self.is_gold[sampled], pl_weight=self.cf.pl_weight, is_augmented=True)
            train_acc = self.evaluator(logits, th.argmax(self.pseudo_labels, 1))
        else:
            loss = self.loss_func(logits[self.train_x], self.gold_labels[self.train_x])
            train_acc = self.evaluator(logits[self.train_x], self.gold_labels[self.train_x])

        loss.backward()
        self.optimizer.step()
        return loss.item(), train_acc

    @th.no_grad()
    def _evaluate(self):
        self.model.eval()
        logits = self._forward(self.g, self.features)
        val_acc = self.evaluator(logits[self.val_x], self.gold_labels[self.val_x])
        test_acc = self.evaluator(logits[self.test_x], self.gold_labels[self.test_x])
        return val_acc, test_acc, logits

    def train(self):
        # ! Training
        for epoch in range(self.cf.epochs):
            t0, es_str = time(), ''
            loss, train_acc = self._train()
            val_acc, test_acc, _ = self._evaluate()
            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
            log_dict = {'Epoch': epoch, 'Time': time() - t0, 'Loss': loss, 'TrainAcc': train_acc, 'ValAcc': val_acc,
                        'ES': es_str, 'GNN_epoch': epoch}
            wandb_dict = {f'{self.wandb_prefix}{k}': v for k, v in log_dict.items() if type(v) in [float, int]}
            wandb_dict.update({f'Step': epoch})
            self.logger.dynamic_log(log_dict, 1 if epoch % LOG_FREQ == 0 else 2, wandb_dict)

        # ! Finished training, load checkpoints
        if self.stopper is not None:
            self.model.load_state_dict(th.load(self.stopper.path))
        return self.model

    @th.no_grad()
    def eval_and_save(self):
        val_acc, test_acc, logits = self._evaluate()
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        if self.cf.model == 'EnGCN':
            pred = self._forward(self.g, self.features)
        else:
            pred = logits

        save_and_report_gnn_result(self.cf, pred, res)
