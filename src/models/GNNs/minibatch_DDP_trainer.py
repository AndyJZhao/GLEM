import dgl.dataloading
from ogb.nodeproppred import Evaluator

import utils.function as uf
from models.GNNs.gnn_utils import *
from models.GNNs.SAGE.model import SAGE
from models.GNNs.SAGE.config import SAGEConfig
from models.GLEM.GLEM_utils import *
from utils.data.datasets import *
from utils.modules.early_stopper import EarlyStopping
import time
import torch.distributed as dist
from utils.data.preprocess import *
LOG_FREQ = 1


class DDP_BatchGNNTrainer():
    def __init__(self, cf: SAGEConfig):
        self.cf, self.logger = cf, cf.logger
        self.log = self.cf.logger.log
        self.wandb_prefix = cf.wandb_prefix if hasattr(cf, 'wandb_prefix') else ''
        th.cuda.set_device(cf.local_rank)
        self.device = cf.device = th.device(cf.local_rank)

        # ! Load data
        self.d = d = cf.data.init()
        d.init_gnn_feature()
        self.g = load_ogb_graph_structure_only(cf)[0]
        if 'ogbInd' in self.cf.dataset:
            self.d.ogb_feat = self.g.ndata['feat']
        self.g = process_graph_structure(self.g, cf)
        # self.g = self.d.g.to(cf.device)
        self.train_x, self.val_x, self.test_x = [
            th.tensor(getattr(d, f'{_}_x'))  for _ in ['train', 'valid', 'test']]
        self.gold_labels = th.from_numpy(self.d['labels']).to(th.int64) #.to(self.device)
        self.is_gold = self.d.is_gold(range(self.d.n_nodes))
        log_graph_feature_source(self.cf)
        # ! EM info init
        if self.cf.is_augmented:
            self.pseudo_labels = self.d.y_hat(range(self.d.n_nodes), on_cpu=True) #.to(self.device)

        #! DDP
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend)
        # ! Trainer init
        if 'ogbInd' in self.cf.dataset:
            self.d.node_feat_dim = self.d.ogb_feat.shape[1]
        if cf.model == 'SAGE':
            self.model = SAGE(self.d.node_feat_dim, cf.n_hidden, cf.data.n_labels, cf.n_layers, F.relu, cf.dropout, batch_size=cf.batch_size, num_workers=cf.num_workers, device=cf.device).to(
                cf.device)
        else:
            ValueError(f'Unimplemented GNNs model {cf.model}!')
        self.model = th.nn.parallel.DistributedDataParallel(self.model, device_ids=[cf.local_rank], output_device=cf.local_rank)

        sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in self.cf.fan_out.split(',')])
        all_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        self.prt_train_dataloader = dgl.dataloading.DataLoader(
            self.g,
            self.train_x,
            sampler,
            device = cf.device,
            use_ddp=True,
            batch_size=self.cf.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.cf.num_workers
        )

        self.train_all_dataloader = dgl.dataloading.DataLoader(
            self.g,
            th.arange(self.g.num_nodes()),
            sampler,
            device = cf.device,
            use_ddp=True,
            batch_size=self.cf.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.cf.num_workers
        )
        self.all_dataloader = dgl.dataloading.DataLoader(
            self.g,
            th.arange(self.g.num_nodes()),
            all_sampler,
            device=cf.device,
            use_ddp=False,
            batch_size=self.cf.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.cf.num_workers)

        self.optimizer = th.optim.Adam(self.model.parameters(), lr=cf.lr, weight_decay=cf.weight_decay)
        self.stopper = EarlyStopping(patience=cf.early_stop, path=cf.checkpoint_file) if cf.early_stop > 0 else None
        self.loss_func = th.nn.CrossEntropyLoss(reduction=cf.ce_reduction)
        self._evaluator = Evaluator(name=cf.data.ogb_name)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels.view(-1, 1)}
        )["acc"]

    def _train(self, epoch):
        import numpy as np
        self.model.train()
        last_loss = []
        last_train_acc = []

        for step, (input_nodes, seeds, blocks) in enumerate(self.train_all_dataloader if self.cf.is_augmented else self.prt_train_dataloader):
            # copy block to gpu
            blocks = [blk.int().to(self.cf.device) for blk in blocks]
            # Load the input features as well as output labels
            batch_size = seeds.shape[0]
            batch_inputs = self.d.node_feature(input_nodes.cpu(), on_cpu=True).to(self.cf.device)
            batch_logits = self.model(blocks, batch_inputs)
            if self.cf.is_augmented:
                batch_labels = self.pseudo_labels[seeds].view(batch_size,-1).to(self.cf.device)
                batch_is_gold = self.d.is_gold(seeds.cpu()).view(-1)
                loss = compute_loss(batch_logits, batch_labels, self.loss_func, batch_is_gold, pl_weight=self.cf.pl_weight, is_augmented=True)
            else:
                batch_labels = self.gold_labels[seeds].to(th.long).to(self.cf.device)
                loss = self.loss_func(batch_logits, batch_labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % self.cf.log_every == 0:
                if self.cf.is_augmented:
                    batch_labels = th.argmax(batch_labels, 1)
                train_acc = self.evaluator(batch_logits, batch_labels)
                if self.cf.local_rank <= 0:
                    gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                    print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | GPU {:.1f} MB'.format(
                epoch, step, loss.item(), train_acc, gpu_mem_alloc))
                last_loss.append(loss.item())
                last_train_acc.append(train_acc)

        return np.mean(last_loss), np.mean(last_train_acc)

    @th.no_grad()
    def _evaluate(self):
        self.model.eval()
        logits = self.model.module.inference(self.g, self.d.node_feature(range(self.d.n_nodes),on_cpu=True).to(self.cf.device), self.all_dataloader)

        val_acc = self.evaluator(logits[self.val_x], self.gold_labels[self.val_x])
        test_acc = self.evaluator(logits[self.test_x], self.gold_labels[self.test_x])

        return val_acc, test_acc, logits

    def train(self):
        # ! Training Loop
        for epoch in range(self.cf.epochs):
            t0, es_str = time.time(), ''
            # Loop over the dataloader to sample the computation dependency graph as a list of
            # blocks.
            loss, train_acc = self._train(epoch)
            if self.cf.local_rank <=0:
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
            else:
                print('eval in local rank 0')
        # ! Finished training, load checkpoints
        if self.stopper is not None and self.cf.local_rank <= 0:
            self.model.load_state_dict(th.load(self.stopper.path))
        else:
            print('wait for local rank 0')
        return self.model

    @th.no_grad()
    def eval_and_save(self):
        if self.cf.local_rank <= 0:
            val_acc, test_acc, logits = self._evaluate()
            res = {'val_acc': val_acc, 'test_acc': test_acc}
            save_and_report_gnn_result(self.cf, logits, res)
        else:
            print('Child local rank finish!')
