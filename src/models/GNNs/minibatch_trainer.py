import dgl.dataloading
from ogb.nodeproppred import Evaluator

import utils.function as uf
from models.GNNs.gnn_utils import *
from models.GNNs.SAGE.model import SAGE
from models.GLEM.GLEM_utils import *
from utils.data.datasets import *
from utils.modules.early_stopper import EarlyStopping
import time
from utils.data.preprocess import *

LOG_FREQ = 1


class BatchGNNTrainer():
    def __init__(self, cf: GNNConfig):
        self.cf, self.logger = cf, cf.logger
        self.log = self.cf.logger.log
        self.wandb_prefix = cf.wandb_prefix if hasattr(cf, 'wandb_prefix') else ''
        log_graph_feature_source(self.cf)
        self.is_ind = 'ind' in cf.dataset
        self.is_aug = self.cf.is_augmented and self.cf.pl_ratio > 0

        # ! Load data
        self.d = d = cf.data.init()
        # self.g = self.d.g.to(cf.device)
        d.init_gnn_feature()  # Global
        self.g = load_ogb_graph_structure_only(cf)[0]
        if 'ogbInd' in self.cf.dataset:
            self.d.ogb_feat = self.g.ndata['feat']
        self.g = process_graph_structure(self.g, cf)  # Local

        self.train_x, self.val_x, self.test_x = [
            th.tensor(getattr(d, f'{_}_x')).to(cf.device) for _ in ['train', 'valid', 'test']]
        # self.labels = th.from_numpy(self.d['labels']).to(th.int64).to(cf.device)
        self.gold_labels = th.from_numpy(self.d['labels']).to(th.int64).to(self.cf.device)
        self.is_gold = self.d.is_gold(range(self.d.n_nodes))
        n_nodes = self.d.n_nodes

        if self.is_ind:
            self.loc2glo = self.g.ndata['_ID'].numpy()
            n_nodes = self.loc2glo.shape[0]
            self.glo2loc = np.zeros(d.n_nodes).astype(np.int)
            self.glo2loc[self.loc2glo] = np.arange(n_nodes)
            # Maps local to global id
        self.nodes = range(n_nodes)
        self.global_id = lambda i: self.loc2glo[i] if self.is_ind else i
        self.local_id = lambda i: self.glo2loc[i] if self.is_ind else i

        # ! Trainer init
        # feat_nums = d.lm_emb_dim + d.n_labels if d.label_as_feat else self.d.lm_emb_dim
        if 'ogbInd' in self.cf.dataset:
            self.d.node_feat_dim = self.d.ogb_feat.shape[1]
        if cf.model == 'SAGE':
            self.model = SAGE(self.d.node_feat_dim, cf.n_hidden, cf.data.n_labels, cf.n_layers, F.relu, cf.dropout, batch_size=cf.batch_size, num_workers=cf.num_workers, device=cf.device).to(
                cf.device)
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print(f'!!!!!The trainable_params is:{trainable_params}')
        else:
            NotImplementedError(f'Unimplemented GNNs model {cf.model}!')

        self.optimizer = th.optim.Adam(self.model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
        self.stopper = EarlyStopping(patience=cf.early_stop, path=cf.checkpoint_file) if cf.early_stop > 0 else None
        self.loss_func = th.nn.CrossEntropyLoss(reduction=cf.ce_reduction)
        self._evaluator = Evaluator(name=cf.data.ogb_name)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels.view(-1, 1)}
        )["acc"]

    def _train(self, epoch, sampler):
        # ! Local: Has to be convert to subgraph ids.
        import numpy as np
        self.model.train()
        last_loss = []
        last_train_acc = []
        if self.is_aug:
            # sample global pseudo label ids for training
            sampled = self.d.get_sampled_aug_ids(int(len(self.d.train_x) * self.cf.pl_ratio))
            sampled = np.concatenate((self.train_x.cpu().numpy(), sampled))
            # map to local_ids
            sampled = th.tensor(self.local_id(sampled)).to(self.cf.device)

        dataloader = dgl.dataloading.DataLoader(
            self.g.to(self.cf.device),
            sampled if self.is_aug else th.tensor(self.local_id(self.train_x.cpu().numpy())).to(self.cf.device),
            sampler,
            batch_size=self.cf.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.cf.num_workers
        )

        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # copy block to gpu
            blocks = [blk.int().to(self.cf.device) for blk in blocks]
            # Load the input features as well as output labels
            batch_size = seeds.shape[0]
            # d is with global index, so have to convert to global id for accessing features.
            batch_inputs = self.d.node_feature(self.global_id(input_nodes))
            batch_labels = self.d.node_labels(self.global_id(seeds)).view(batch_size, -1)
            batch_is_gold = self.d.is_gold(self.global_id(seeds)).view(-1)
            # Compute loss and prediction
            batch_logits = self.model(blocks, batch_inputs)
            loss = compute_loss(batch_logits, batch_labels, self.loss_func, batch_is_gold, pl_weight=self.cf.pl_weight, is_augmented=True)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % self.cf.log_every == 0:
                batch_labels = th.argmax(batch_labels, 1)
                train_acc = self.evaluator(batch_logits, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, loss.item(), train_acc, gpu_mem_alloc))
                last_loss.append(loss.item())
                last_train_acc.append(train_acc)

        return np.mean(last_loss), np.mean(last_train_acc)

    def _inference(self):
        # ! Global Eval and save
        logits = self.model.inference(self.g.to(self.cf.device), self.d.node_feature(self.global_id(self.nodes)), self.cf.device)  # small-graph
        if self.is_ind:
            # Inductive: insert subgraph logits to full graph
            global_logits = th.zeros(self.d.n_nodes, logits.shape[1]).to(self.cf.device)
            global_logits[self.g.ndata['_ID']] = logits
            return global_logits
        else:
            return logits

    @th.no_grad()
    def _evaluate(self):
        # Global ids
        self.model.eval()
        logits = self._inference()
        val_acc = self.evaluator(logits[self.val_x], self.gold_labels[self.val_x])
        test_acc = self.evaluator(logits[self.test_x], self.gold_labels[self.test_x])

        return val_acc, test_acc, logits

    def train(self):
        # ! Prepare Dataloader What sampler should be used?
        sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in self.cf.fan_out.split(',')])

        # th.arange(g.num_nodes()).to(self.cf.device)
        # ! Training Loop
        for epoch in range(self.cf.epochs):
            t0, es_str = time.time(), ''
            # Loop over the dataloader to sample the computation dependency graph as a list of
            # blocks.
            loss, train_acc = self._train(epoch, sampler)
            val_acc, test_acc, _ = self._evaluate()
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
        if self.stopper is not None:
            self.model.load_state_dict(th.load(self.stopper.path))
        return self.model

    @th.no_grad()
    def eval_and_save(self):
        val_acc, test_acc, logits = self._evaluate()
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        save_and_report_gnn_result(self.cf, logits, res)
