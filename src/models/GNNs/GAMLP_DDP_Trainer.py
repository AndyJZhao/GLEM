import gc
import os.path

import numpy as np
from ogb.nodeproppred import Evaluator
from models.GNNs.GAMLP_DDP.model import R_GAMLP_RLU
from models.GNNs.GAMLP_DDP.config import GAMLP_DDP_Config
from models.GNNs.gnn_utils import *
from models.GLEM.GLEM_utils import *
from utils.data.datasets import *
from utils.modules.early_stopper import EarlyStopping
import torch.distributed as dist
from utils.data.preprocess import *
from bidict import bidict

LOG_FREQ = 10
import time


class GAMLP_DDP_Trainer():
    def __init__(self, cf: GAMLP_DDP_Config):
        self.cf, self.logger = cf, cf.logger
        self.log = self.cf.logger.log
        self.wandb_prefix = cf.wandb_prefix if hasattr(cf, 'wandb_prefix') else ''
        th.cuda.set_device(cf.local_rank)
        self.is_ind = 'ind' in cf.dataset
        self.is_ddp = min(th.cuda.device_count(), len(cf.gpus.split(','))) > 1 and cf.gpus != '-1'
        if self.cf.process_feat_only:
            # ! Load data
            # uf.remove_file(SeqGraph(cf)._processed_flag['g_info'])
            self.d = d = cf.data.init()
            raw_g = load_ogb_graph_structure_only(cf)[0]
            if 'ogbInd' in self.cf.dataset:
                self.d.ogb_feat = raw_g.ndata['feat'].cpu()
            self.g = process_graph_structure(raw_g, cf)
            log_graph_feature_source(self.cf)
            if self.is_ind:
                pickle_save(self.g.ndata['_ID'].numpy(), cf._g_id_file)
            if not cf.is_processed:
                # ! Prepare features on local rank 0
                # ! Load full-graph
                print(f'Processing features on LOCAL_RANK #{cf.local_rank}...')
                d.init_gnn_feature()

                features = self.d.node_feature(range(self.d.n_nodes), on_cpu=True)
                from models.GNNs.GAMLP_DDP.utils.load_dataset import neighbor_average_features
                _ = neighbor_average_features(self.g, features, cf)
                uf.pickle_save('processed', cf.processed_file)
                print(f'Feature processing finished on LOCAL_RANK #{cf.local_rank}')
                del features
                gc.collect()
            else:
                print(f'Found preprocessed feature {cf.processed_file}')
            # ! Prepare Label_emb
            self.gold_labels = th.from_numpy(self.d['labels']).to(th.int64)  # .to(self.device)
            self.train_x, self.val_x, self.test_x = [
                th.tensor(getattr(d, f'{_}_x')) for _ in ['train', 'valid', 'test']]
            from models.GNNs.GAMLP_DDP.utils.load_dataset import prepare_label_emb
            if self.cf.is_augmented:
                self.pseudo_labels = self.d.y_hat(range(self.d.n_nodes), on_cpu=True)  # .to(self.device)
                if self.cf.average == 'T0':
                    self.label_emb = prepare_label_emb(cf.label_num_hops, self.g, self.pseudo_labels, cf.data.n_labels, cf, self.train_x, self.val_x,
                                                       self.test_x)
                elif self.cf.average == 'T1':
                    self.label_emb = prepare_label_emb(cf.label_num_hops, self.g, self.gold_labels, cf.data.n_labels,
                                                       cf, self.train_x, self.val_x, self.test_x)
                else:
                    self.label_emb = self.pseudo_labels

            else:
                self.label_emb = prepare_label_emb(cf.label_num_hops, self.g, self.gold_labels, cf.data.n_labels, cf,
                                                   self.train_x, self.val_x, self.test_x)
            return

        self.device = cf.device = th.device(max(0, cf.local_rank)) if cf.gpus != '-1' else th.device('cpu')
        self.d = d = cf.data.init()
        self.train_x, self.val_x, self.test_x = [
            th.tensor(getattr(d, f'{_}_x')) for _ in ['train', 'valid', 'test']]
        self.is_gold = self.d.is_gold(range(self.d.n_nodes))
        self.gold_labels = th.from_numpy(self.d['labels']).to(th.int64)  # .to(self.device)
        n_nodes = self.d.n_nodes

        if self.is_ind:
            self.loc2glo = pickle_load(cf._g_id_file)
            n_nodes = self.loc2glo.shape[0]
            self.glo2loc = np.zeros(d.n_nodes).astype(np.int)
            self.glo2loc[self.loc2glo] = np.arange(n_nodes)
            # Maps local to global id
        nodes = range(n_nodes)

        self.global_id = lambda i: self.loc2glo[i] if self.is_ind else i
        self.local_id = lambda i: self.glo2loc[i] if self.is_ind else i

        if 'ogbInd' in self.cf.dataset:
            raw_g = load_ogb_graph_structure_only(cf)[0]
            self.d.ogb_feat = raw_g.ndata['feat'].cpu()
            self.d.node_feat_dim = self.d.ogb_feat.shape[1]
        self.feats = [np.memmap(os.path.join('temp', cf.processed_feat(_)), mode='r', dtype=np.float16, shape=(n_nodes, self.d.node_feat_dim)) for _ in range(cf.num_hops + 1)]
        self.label_emb = np.memmap(os.path.join('temp', cf.processed_label), mode='r', dtype=np.float16, shape=(n_nodes, self.d.n_labels))

        if self.cf.is_augmented:
            self.pseudo_labels = self.d.y_hat(range(self.d.n_nodes), on_cpu=True)  # .to(self.device)

        if self.is_ddp:
            dist_backend = 'nccl'
            dist.init_process_group(backend=dist_backend)

        # ! Data parallel
        if cf.model == 'GAMLP_DDP':
            if 'ogbInd' in self.cf.dataset:
                self.d.node_feat_dim = self.d.ogb_feat.shape[1]
            self.model = R_GAMLP_RLU(self.d.node_feat_dim, cf.n_hidden, cf.data.n_labels, cf.num_hops + 1, cf.dropout, cf.input_dropout, cf.att_dropout, cf.label_drop, cf.alpha, cf.n_layers,
                                     cf.n_layers_3, cf.act, cf.pre_process, cf.residual,
                                     cf.pre_dropout, cf.bns, input_norm=cf.input_norm == 'T')
        else:
            ValueError(f'Unimplemented GNNs model {cf.model}!')

        if self.is_ddp:
            self.model = th.nn.parallel.DistributedDataParallel(self.model.cuda(), device_ids=[cf.local_rank], output_device=cf.local_rank)
        else:
            self.model = self.model.to(self.device)
        if self.is_ddp:
            train_sampler = th.utils.data.distributed.DistributedSampler(self.train_x)
            pl_sampler = th.utils.data.distributed.DistributedSampler(self.d.pl_nodes)
        else:
            train_sampler, pl_sampler = None, None
        if cf.is_augmented:
            tr_bsz = int(min(cf.aug_batch_size, len(self.train_x)) / len(cf.gpus))
        else:  # Pretrain
            tr_bsz = int(min(cf.prt_batch_size, len(self.train_x)) / len(cf.gpus))

        self.train_loader = th.utils.data.DataLoader(self.train_x, sampler=train_sampler, batch_size=tr_bsz, drop_last=False, num_workers=1, pin_memory=True)
        pl_bsz = max(1, int(tr_bsz * self.cf.pl_ratio))
        # print(f"TR bsz = {tr_bsz}");print(f"PL bsz = {pl_bsz}")

        self.pl_loader = th.utils.data.DataLoader(
            # self.d.pl_nodes, sampler=pl_sampler, batch_size=int(tr_bsz * self.cf.pl_ratio),
            self.d.pl_nodes, sampler=pl_sampler, batch_size=pl_bsz,
            drop_last=False, num_workers=1, pin_memory=True)

        self.all_visible_nodes_loader = th.utils.data.DataLoader(
            nodes, batch_size=cf.eval_batch_size,
            drop_last=False, num_workers=1, pin_memory=True)

        self.val_loader = th.utils.data.DataLoader(self.val_x, batch_size=cf.eval_batch_size, shuffle=False, drop_last=False)
        self.test_loader = th.utils.data.DataLoader(self.test_x, batch_size=cf.eval_batch_size, shuffle=False, drop_last=False)

        # ! Trainer init
        self.optimizer = th.optim.Adam(self.model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
        self.stopper = EarlyStopping(patience=cf.early_stop, path=cf.checkpoint_file) if cf.early_stop > 0 else None
        self.loss_func = th.nn.CrossEntropyLoss(reduction=cf.ce_reduction).cuda(cf.local_rank)
        self._evaluator = Evaluator(name=cf.data.ogb_name)

        self.evaluator = lambda preds, labels: self._evaluator.eval({
            "y_true": labels.view(-1, 1),
            "y_pred": preds.view(-1, 1),
        })["acc"]

    def _get_batch_label_emb(self, nodes):
        nodes = self.local_id(nodes)
        return th.from_numpy(np.array(self.label_emb[nodes])).to(th.float32).to(self.device)

    def _get_batch_feat_train(self, nodes):
        nodes = self.local_id(nodes)
        return [th.from_numpy(np.array(x[nodes])).to(th.float32).to(self.device) for x in self.feats]

    def _train(self, epoch):
        # ! Shared
        self.model.train()
        total_loss = 0
        iter_num = 0
        y_true = []
        y_pred = []
        if self.is_ddp:
            self.train_loader.sampler.set_epoch(epoch)
            self.pl_loader.sampler.set_epoch(epoch)
        for batch_data in zip(self.train_loader, self.pl_loader):
            batch, pl_nodes = batch_data
            # print(f'Local RANK={self.cf.local_rank} Batch shape {batch.shape} PL nodes shape {pl_nodes.shape}')
            # assert self.d.is_gold(pl_nodes).sum() == 0
            if self.cf.is_augmented:
                batch = th.cat([batch, pl_nodes])
            batch_feats = self._get_batch_feat_train(batch)  # local
            label_emb = self._get_batch_label_emb(batch)  # local
            output_att = self.model(batch_feats, label_emb.to(self.cf.device))
            y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())  #
            if self.cf.is_augmented and self.cf.pl_ratio > 0:
                y_true.append(self.gold_labels[batch].to(th.long).cpu())
                # y_true.append(self.pseudo_labels[batch].to(th.long))
                l1 = compute_loss(output_att, self.pseudo_labels[batch].to(self.device), self.loss_func, self.d.is_gold(batch).view(-1), pl_weight=self.cf.pl_weight, is_augmented=True)
            else:
                y_true.append(self.gold_labels[batch].to(th.long).cpu())
                l1 = self.loss_func(output_att, self.gold_labels[batch].to(self.device))
            loss_train = l1
            total_loss += loss_train
            self.optimizer.zero_grad()
            l1.backward()
            self.optimizer.step()
            iter_num += 1
        iter_num = 1 if iter_num == 0 else iter_num
        loss = total_loss / iter_num
        # if self.cf.is_augmented:
        #     y_true = [_.argmax(-1) for _ in y_true]
        train_acc = self.evaluator(th.cat(y_true, dim=0), th.cat(y_pred, dim=0))

        return loss.item(), train_acc

    @th.no_grad()
    def _evaluate(self, test_loader):
        self.model.eval()
        preds_test = []
        true_test = []

        for batch in test_loader:
            batch_feats = self._get_batch_feat_train(batch)
            label_emb = self._get_batch_label_emb(batch)
            preds_test.append(th.argmax(self.model(batch_feats, label_emb.to(self.device)), dim=-1))
            true_test.append(self.gold_labels[batch])
        true_test = th.cat(true_test)
        preds_test = th.cat(preds_test, dim=0)
        test_acc = self.evaluator(true_test, preds_test)

        return test_acc

    def train(self):
        if self.cf.process_feat_only:
            return
        for epoch in range(self.cf.epochs):
            t0, es_str = time.time(), ''
            loss, train_acc = self._train(epoch)
            # if self.cf.local_rank <= 0:
            val_acc = self._evaluate(self.val_loader)
            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
            log_dict = {'Epoch': epoch, 'Time': time.time() - t0, 'Loss': loss, 'TrainAcc': train_acc, 'ValAcc': val_acc,
                        'ES': es_str, 'GNN_epoch': epoch}
            wandb_dict = {f'{self.wandb_prefix}{k}': v for k, v in log_dict.items() if type(v) in [float, int]}
            wandb_dict.update({f'Step': epoch})
            self.logger.dynamic_log(log_dict, 1 if epoch % LOG_FREQ == 0 else 2, wandb_dict)

        # ! Finished training, load checkpoints
        if self.cf.local_rank <= 0 and self.stopper is not None:
            self.model.load_state_dict(th.load(self.stopper.path))
        th.cuda.empty_cache()
        return self.model

    @th.no_grad()
    def eval_and_save(self):
        if self.cf.process_feat_only:
            return
        if self.cf.local_rank <= 0:
            val_acc = self._evaluate(self.val_loader)
            test_acc = self._evaluate(self.test_loader)
            res = {'val_acc': val_acc, 'test_acc': test_acc}
            if self.cf.model == 'GAMLP_DDP':
                from models.GNNs.GAMLP_DDP.utils.utils import gen_output_torch
                print('\n\nStart inference')
                logits = gen_output_torch(self.model, self.feats, self.all_visible_nodes_loader, self.device, self.label_emb)
                if self.is_ind:
                    pred = th.zeros(self.d.n_nodes, logits.shape[1]).to(logits.device)
                    pred[self.loc2glo] = logits
                else:
                    pred = logits
            else:
                pred = 'None'
                ValueError(f'Unimplemented GNNs model {self.cf.model}!')

            save_and_report_gnn_result(self.cf, pred, res)
        else:
            print('Wating for local rank 0')
