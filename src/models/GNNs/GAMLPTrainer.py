from time import time

from ogb.nodeproppred import Evaluator
from models.GNNs.GAMLP.model import R_GAMLP_RLU
from models.GNNs.GAMLP.config import GAMLPConfig
from models.GNNs.gnn_utils import *
from models.GLEM.GLEM_utils import *
from utils.data.datasets import *
from utils.modules.early_stopper import EarlyStopping
from utils.data.preprocess import *
import os.path

LOG_FREQ = 10
from time import time

class GAMLP_Trainer():
    def __init__(self, cf: GAMLPConfig):
        self.cf, self.logger = cf, cf.logger
        self.log = self.cf.logger.log
        self.wandb_prefix = cf.wandb_prefix if hasattr(cf, 'wandb_prefix') else ''

        # ! Load data
        self.d = d = cf.data.init()
        d.init_gnn_feature()
        self.g = load_ogb_graph_structure_only(cf)[0]
        self.g = process_graph_structure(self.g, cf)
        self.train_x, self.val_x, self.test_x = [
            th.tensor(getattr(d, f'{_}_x')) for _ in ['train', 'valid', 'test']]

        # ! Prepare features:
        from models.GNNs.GAMLP.utils.load_dataset import prepare_features # neighbor_average_features
        features = self.d.node_feature(range(self.d.n_nodes), on_cpu=True)

        save_memmap(features, os.path.join('temp', cf.processed_feat(0)), np.float16)
        _ = prepare_features(self.g, features, cf)
        self.features = [np.memmap(os.path.join('temp', cf.processed_feat(_)), mode='r', dtype=np.float16, shape=(self.d.n_nodes, self.d.node_feat_dim)) for _ in range(cf.num_hops + 1)]

        self.gold_labels = th.from_numpy(self.d['labels']).to(th.int64)
        self.is_gold = self.d.is_gold(range(self.d.n_nodes))
        if self.cf.is_augmented:
            self.pseudo_labels = self.d.y_hat(range(self.d.n_nodes), on_cpu=True)
        log_graph_feature_source(self.cf)
        # Prepare Neverage label information
        from models.GNNs.GAMLP.utils.load_dataset import prepare_label_emb
        if self.cf.is_augmented:
            if self.cf.average == 'T0':
                self.label_emb = prepare_label_emb(cf.label_num_hops, self.g, self.pseudo_labels,cf.data.n_labels, self.train_x, self.val_x, self.test_x)
            elif self.cf.average == 'T1':
                self.label_emb = prepare_label_emb(cf.label_num_hops, self.g, self.gold_labels, cf.data.n_labels,
                                                   self.train_x, self.val_x, self.test_x)
            else:
                self.label_emb = self.pseudo_labels

        else:
            self.label_emb = prepare_label_emb(cf.label_num_hops, self.g, self.gold_labels, cf.data.n_labels,
                                              self.train_x, self.val_x,self.test_x)

        # ! Trainer init
        if cf.model == 'GAMLP':
            self.model = R_GAMLP_RLU(self.d.node_feat_dim, cf.n_hidden, cf.data.n_labels, cf.num_hops + 1, cf.dropout, cf.input_dropout, cf.att_dropout, cf.label_drop, cf.alpha, cf.n_layers, cf.n_layers_3, cf.act, cf.pre_process, cf.residual,
                                 cf.pre_dropout, cf.bns, input_norm = cf.input_norm == 'T').to(cf.device)
        else:
            ValueError(f'Unimplemented GNNs model {cf.model}!')

        # 准备loader，用于每个epoch中
        self.train_loader = th.utils.data.DataLoader(self.train_x, batch_size=cf.prt_batch_size, shuffle=True, drop_last=False)

        #! Prepare enhance pseudolabels
        pl_bsz = int(cf.prt_batch_size * cf.pl_ratio)
        self.pl_loader = th.utils.data.DataLoader(
            # self.d.pl_nodes, sampler=pl_sampler, batch_size=int(tr_bsz * self.cf.pl_ratio),
            self.d.pl_nodes, batch_size=pl_bsz,drop_last=False, num_workers=1, pin_memory=True)

        # self.aug_train_loader = th.utils.data.DataLoader(range(self.d.n_nodes), batch_size=cf.prt_batch_size, shuffle=True, drop_last=False)

        self.all_eval_loader = th.utils.data.DataLoader(range(self.d.n_nodes), batch_size=cf.eval_batch_size,shuffle=False, drop_last=False)

        self.val_loader = th.utils.data.DataLoader(self.val_x, batch_size=cf.eval_batch_size, shuffle=False, drop_last=False)

        self.test_loader = th.utils.data.DataLoader(self.test_x, batch_size=cf.eval_batch_size,shuffle=False, drop_last=False)

        self.optimizer = th.optim.Adam(self.model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
        self.stopper = EarlyStopping(patience=cf.early_stop, path=cf.checkpoint_file) if cf.early_stop > 0 else None
        self.loss_func = th.nn.CrossEntropyLoss(reduction=cf.ce_reduction)
        self._evaluator = Evaluator(name=cf.data.ogb_name)


        # self.evaluator = lambda pred, labels: self._evaluator.eval(
        #     {"y_true": labels.view(-1, 1), "y_pred": pred.view(-1, 1)}
        # )["acc"]

        self.evaluator = lambda preds, labels: self._evaluator.eval({
            "y_true": labels.view(-1, 1),
            "y_pred": preds.view(-1, 1),
        })["acc"]

    def _get_batch_feat_train(self, nodes):
        return [th.from_numpy(np.array(x[nodes])).to(th.float32).to(self.cf.device) for x in self.features]


    def _train(self):
        # ! Shared
        self.model.train()
        total_loss = 0
        iter_num = 0
        y_true = []
        y_pred = []
        for batch_data in zip(self.train_loader, self.pl_loader):
            batch, pl_nodes = batch_data
            if self.cf.is_augmented:
                batch = th.cat([batch, pl_nodes])
            batch_feats = self._get_batch_feat_train(batch) #[x[batch].to(self.cf.device) for x in self.features]
            output_att = self.model(batch_feats, self.label_emb[batch].to(self.cf.device))
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
            if self.cf.is_augmented and self.cf.pl_ratio > 0:
                y_true.append(self.gold_labels[batch].to(th.long).cpu())
                l1 = compute_loss(output_att, self.pseudo_labels[batch].to(self.cf.device), self.loss_func, self.d.is_gold(batch).view(-1), pl_weight=self.cf.pl_weight, is_augmented=True)
            else:
                y_true.append(self.gold_labels[batch].to(th.long).cpu())
                l1 = self.loss_func(output_att, self.gold_labels[batch].to(self.cf.device))
            loss_train = l1
            total_loss += loss_train
            self.optimizer.zero_grad()
            l1.backward()
            self.optimizer.step()
            iter_num += 1
        loss = total_loss / iter_num
        train_acc = self.evaluator(th.cat(y_true, dim=0), th.cat(y_pred, dim=0))

        return loss.item(), train_acc

    @th.no_grad()
    def _evaluate(self, test_loader):
        self.model.eval()
        preds_test = []
        true_test = []

        for batch in test_loader:
            batch_feats = self._get_batch_feat_train(batch)
            preds_test.append(th.argmax(self.model(batch_feats, self.label_emb[batch].to(self.cf.device)), dim=-1))
            true_test.append(self.gold_labels[batch])
        true_test = th.cat(true_test)
        preds_test = th.cat(preds_test, dim=0)
        test_acc = self.evaluator(true_test, preds_test)

        return test_acc

    def train(self):
        # ! Training
        for epoch in range(self.cf.epochs):
            import gc
            gc.collect()
            t0, es_str = time(), ''
            loss, train_acc = self._train()
            val_acc = self._evaluate(self.val_loader)
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
        val_acc = self._evaluate(self.val_loader)
        test_acc = self._evaluate(self.test_loader)
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        if self.cf.model == 'GAMLP':
            from models.GNNs.GAMLP.utils.utils import gen_output_torch
            pred = gen_output_torch(self.model, self.features, self.all_eval_loader, self.cf.device, self.label_emb)
        else:
            pred = 'None'
            ValueError(f'Unimplemented GNNs model {self.cf.model}!')

        save_and_report_gnn_result(self.cf, pred, res)
