from time import time

from ogb.nodeproppred import Evaluator
from models.GNNs.SAGN.model import SAGN,SLEModel,EMA
from models.GNNs.SAGN.utils.layer import GroupMLP
from models.GNNs.SAGN.config import SAGN_Config
from models.GNNs.gnn_utils import *
from models.GraphVF.gvf_utils import *
from utils.data.datasets import *
from utils.modules.early_stopper import EarlyStopping
from utils.data.preprocess import *

LOG_FREQ = 10
from time import time

class SAGN_Trainer():
    def __init__(self, cf: SAGN_Config):
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

        # ! Prepare featurs:
        from models.GNNs.SAGN.utils.load_dataset import prepare_features
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
        from models.GNNs.SAGN.utils.load_dataset import prepare_label_emb
        if self.cf.is_augmented:
            self.label_emb = prepare_label_emb(cf.label_num_hops, self.g, self.pseudo_labels,cf.data.n_labels, self.train_x, self.val_x, self.test_x)
        else:
            self.label_emb = prepare_label_emb(cf.label_num_hops, self.g, self.gold_labels, cf.data.n_labels,
                                              self.train_x, self.val_x,self.test_x)

        # ! Trainer init
        if cf.model == 'SAGN':
            label_in_feats = self.label_emb.shape[1] if self.label_emb is not None else self.cf.data.n_labels
            self.model = self.gen_model_sagn(label_in_feats=label_in_feats).to(cf.device)
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print(f'!!!!!GNN Phase, trainable_params are {trainable_params}')
            # self.model = self.model.to(cf.device)
            if cf.mean_teacher == True:
                print("use teacher-SAGN")
                label_in_feats = self.label_emb.shape[1] if self.label_emb is not None else self.cf.data.n_labels
                self.teacher_model = self.gen_model_sagn(label_in_feats=label_in_feats)
                self.teacher_model = self.teacher_model.to(cf.device)
                for param in self.teacher_model.parameters():
                    param.detach_()
        else:
            ValueError(f'Unimplemented GNNs model {cf.model}!')
        if cf.ema == True:
            print("use ema")
            self.ema = EMA(self.model, cf.decay)
            self.ema.register()
        else:
            self.ema = None

        # 准备loader，用于每个epoch中
        self.train_loader = th.utils.data.DataLoader(self.train_x, batch_size=cf.batch_size, shuffle=True, drop_last=False)
        pl_bsz = int(cf.batch_size * cf.pl_ratio)
        self.pl_loader = th.utils.data.DataLoader(self.d.pl_nodes, batch_size=pl_bsz, drop_last=False, num_workers=1, pin_memory=True)
        # self.aug_train_loader = th.utils.data.DataLoader(range(self.d.n_nodes), batch_size=cf.batch_size, shuffle=True,
        #                                              drop_last=False)
        self.all_eval_loader = th.utils.data.DataLoader(range(self.d.n_nodes), batch_size=cf.eval_batch_size,shuffle=False, drop_last=False)

        self.val_loader = th.utils.data.DataLoader(self.val_x, batch_size=cf.eval_batch_size, shuffle=False, drop_last=False)

        self.test_loader = th.utils.data.DataLoader(self.test_x, batch_size=cf.eval_batch_size,shuffle=False, drop_last=False)

        self.optimizer = th.optim.Adam(self.model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
        self.stopper = EarlyStopping(patience=cf.early_stop, path=cf.checkpoint_file) if cf.early_stop > 0 else None
        self.loss_func = th.nn.CrossEntropyLoss(reduction=cf.ce_reduction)
        self._evaluator = Evaluator(name=cf.data.ogb_name)

        self.evaluator = lambda preds, labels: self._evaluator.eval({
            "y_true": labels.view(-1, 1),
            "y_pred": preds.view(-1, 1),
        })["acc"]

    def _get_batch_feat_train(self, nodes):
        return [th.from_numpy(np.array(x[nodes])).to(th.float32).to(self.cf.device) for x in self.features]

    def _train(self):
        # ! train_sagn
        self.model.train()
        total_loss = 0
        iter_num = 0
        y_true = []
        y_pred = []
        #for batch in self.aug_train_loader if self.cf.is_augmented else self.train_loader:
        for batch_data in zip(self.train_loader, self.pl_loader):
            batch, pl_nodes = batch_data
            if self.cf.is_augmented:
                batch = th.cat([batch, pl_nodes])
            batch_feats = self._get_batch_feat_train(batch)
            if self.label_emb is not None:
                batch_label_emb = self.label_emb[batch].to(self.cf.device)
            else:
                batch_label_emb = None
            output_att, _ = self.model(batch_feats, batch_label_emb)
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())  #
            if self.cf.is_augmented and self.cf.pl_ratio > 0:
                y_true.append(self.gold_labels[batch].to(th.long).cpu())
                l1 = compute_loss(output_att, self.pseudo_labels[batch].to(self.cf.device), self.loss_func, self.d.is_gold(batch).view(-1), pl_weight=self.cf.pl_weight, is_augmented=True)
            else:
                y_true.append(self.gold_labels[batch].to(th.long))
                l1 = self.loss_func(output_att, self.gold_labels[batch].to(self.cf.device))
            loss_train = l1
            total_loss += loss_train
            self.optimizer.zero_grad()
            loss_train.backward()
            self.optimizer.step()
            iter_num += 1
            if self.ema != None:
                self.ema.update()
        loss = total_loss / iter_num
        train_acc = self.evaluator(th.cat(y_true, dim=0), th.cat(y_pred, dim=0))

        return loss.item(), train_acc

    def _train_mean_teacher(self, train_loader, enhance_loader_cons):
        # ! train_sagn_mean_teacher
        self.model.train()
        total_loss = 0
        total_loss_mse = 0
        total_loss_kl = 0
        total_loss_supervised = 0
        iter_num = 0
        y_true = []
        y_pred = []
        for idx1, idx2 in zip(train_loader, enhance_loader_cons):
            if len(idx1) == 1:
                continue
            batch_feats = self._get_batch_feat_train(idx1)
            batch_feats_cons = self._get_batch_feat_train(idx2)
            if self.label_emb is not None:
                batch_label_emb = self.label_emb[idx1].to(self.cf.device)
                batch_label_emb_cons = self.label_emb[idx2].to(self.cf.device)
            else:
                batch_label_emb = None
                batch_label_emb_cons = None

            out, _ = self.model(batch_feats, batch_label_emb)
            out_s, _ = self.model(batch_feats_cons, batch_label_emb_cons)
            out_t, _ = self.teacher_model(batch_feats_cons, batch_label_emb_cons)
            y_pred.append(out.argmax(dim=-1, keepdim=True).cpu())
            y_true.append(self.gold_labels[idx1].to(th.long))

            from SAGN.utils.utils import consis_loss_mean_teacher
            mse, kl = consis_loss_mean_teacher(out_t, out_s, self.cf.tem, self.cf.lam)
            kl = kl * self.cf.kl_lam

            L1 = self.loss_func(out, self.gold_labels[idx1].to(self.cf.device))
            if self.cf.kl == False:
                loss = L1 + mse
            else:
                loss = L1 + kl

            total_loss += loss
            total_loss_mse += mse
            total_loss_kl += kl
            total_loss_supervised += L1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.ema != None:
                self.ema.update()

            alpha = self.cf.ema_decay
            for mean_param, param in zip(self.teacher_model.parameters(), self.model.parameters()):
                mean_param.data.mul_(alpha).add_(1 - alpha, param.data)
            iter_num += 1
        loss = total_loss / iter_num
        loss_mse = total_loss_mse / iter_num
        loss_sup = total_loss_supervised / iter_num
        loss_kl = total_loss_kl / iter_num
        train_acc = self.evaluator(th.cat(y_true, dim=0), th.cat(y_pred, dim=0))

        return loss, train_acc


    @th.no_grad()
    def _evaluate(self, val_loader):
        self.model.eval()
        preds_val = []
        true_val = []
        for batch in val_loader:
            batch_feats = self._get_batch_feat_train(batch)
            output, _ = self.model(batch_feats, self.label_emb[batch].to(self.cf.device))
            preds_val.append(th.argmax(output, dim=-1))
            true_val.append(self.gold_labels[batch])
        true_val = th.cat(true_val)
        preds_val = th.cat(preds_val, dim=0)
        val_acc = self.evaluator(preds_val, true_val)

        return val_acc

    def gen_model_sagn(self, label_in_feats):
        num_hops = self.cf.num_hops + 1
        base_model = SAGN(self.d.node_feat_dim, self.cf.n_hidden, self.cf.data.n_labels, num_hops,
                          self.cf.mlp_layer, self.cf.num_heads,
                          weight_style=self.cf.weight_style,
                          dropout=self.cf.dropout,
                          input_drop=self.cf.input_drop,
                          attn_drop=self.cf.att_drop,
                          zero_inits=self.cf.zero_inits,
                          position_emb=self.cf.position_emb,
                          focal=self.cf.focal)
        label_model = GroupMLP(label_in_feats,
                               self.cf.n_hidden,
                               self.cf.data.n_labels,
                               self.cf.num_heads,
                               self.cf.label_mlp_layer,
                               self.cf.label_drop,
                               residual=self.cf.label_residual, )
        model = SLEModel(base_model, label_model)
        return model

    def train(self):
        # ! Training
        for epoch in range(self.cf.epochs):
            t0, es_str = time(), ''
            if self.cf.mean_teacher == False:
                loss, train_acc = self._train()
            else:
                if epoch < (self.cf.warm_up + 1): #1000 100
                    loss, train_acc = self._train() #EM-Iter LM Pesudo-labels
                else:
                    if epoch == (self.cf.warm_up + 1):
                        print("start mean teacher")
                    if (epoch - 1) % self.cf.gap == 0 or epoch == (self.cf.warm_up + 1):
                        from models.GNNs.SAGN.utils.utils import gen_output_torch
                        preds = gen_output_torch(self.model, self.features, self.all_eval_loader, self.cf.device, self.label_emb, self.ema)
                        predict_prob = preds.softmax(dim=1)

                        threshold = self.cf.top - (self.cf.top - self.cf.down) * epoch / self.cf.epochs

                        enhance_idx_cons = th.arange(self.d.n_nodes)[(predict_prob.max(1)[0] > threshold) & ~self.d.gi.is_gold]
                        enhance_loader_cons = th.utils.data.DataLoader(enhance_idx_cons, batch_size=int(
                            self.cf.batch_size * len(enhance_idx_cons) / (len(enhance_idx_cons) + self.train_x.shape[0])),
                                                                          shuffle=True, drop_last=False)
                        train_loader_with_pseudos = th.utils.data.DataLoader(self.train_x, batch_size= int(self.cf.batch_size * self.train_x.shape[0] / (len(enhance_idx_cons) + self.train_x.shape[0])) ,
                                                                             shuffle=True, drop_last=False)

                    loss, train_acc = self._train_mean_teacher(train_loader_with_pseudos, enhance_loader_cons)
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
        if self.cf.model == 'SAGN':
            from models.GNNs.SAGN.utils.utils import gen_output_torch
            pred = gen_output_torch(self.model, self.features, self.all_eval_loader, self.cf.device, self.label_emb, self.ema)
        else:
            pred = 'None'
            ValueError(f'Unimplemented GNNs model {self.cf.model}!')

        save_and_report_gnn_result(self.cf, pred, res)
