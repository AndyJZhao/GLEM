import os
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

import numpy as np
import torch
import torch_geometric.datasets
from sklearn.metrics import f1_score
from torch.profiler import ProfilerActivity, profile
from torch_geometric.transforms import ToSparseTensor
from Precomputing.Ensembling.EnGCN import EnGCN

def load_data(dataset_name, to_sparse=True):
    if dataset_name in ["ogbn-products", "ogbn-papers100M", "ogbn-arxiv"]:
        root = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "dataset", dataset_name
        )
        T = ToSparseTensor() if to_sparse else lambda x: x
        dataset = PygNodePropPredDataset(name=dataset_name, root=root, transform=T)
        processed_dir = dataset.processed_dir
        split_idx = dataset.get_idx_split()
        evaluator = Evaluator(name=dataset_name)
        data = dataset[0]
        split_masks = {}
        for split in ["train", "valid", "test"]:
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[split_idx[split]] = True
            data[f"{split}_mask"] = mask
            split_masks[f"{split}"] = data[f"{split}_mask"]

        x = data.x
        y = data.y = data.y.squeeze()

    else:
        raise Exception(f"the dataset of {dataset} has not been implemented")
    return data, x, y, split_masks, evaluator, processed_dir



class trainer(object):
    def __init__(self, args):

        self.dataset = args.dataset
        self.device = torch.device(f"cuda:{args.cuda_num}" if args.cuda else "cpu")
        self.args = args
        self.args.device = self.device

        self.type_model = args.type_model
        self.epochs = args.epochs
        self.eval_steps = args.eval_steps

        # used to indicate multi-label classification.
        # If it is, using BCE and micro-f1 performance metric
        self.multi_label = args.multi_label
        if self.multi_label:
            self.loss_op = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_op = torch.nn.NLLLoss()

        (
            self.data,
            self.x,
            self.y,
            self.split_masks,
            self.evaluator,
            self.processed_dir,
        ) = load_data(args.dataset, args.tosparse)

        #! load emb from LM
        if args.LM_emb_path != None:
            self.x = torch.from_numpy(np.array(np.memmap(args.LM_emb_path, mode='r', dtype=np.float16, shape=(2449029,768)))).to(torch.float32)
            print('load from GLEM:LM!')
        elif args.GIANT != None:
            self.x = torch.tensor(np.load(args.GIANT)).float()
        else:
            print('load from OGB feature!')

        if self.type_model == "EnGCN":
            self.model = EnGCN(
                args,
                self.data,
                self.evaluator,
            )
        else:
            raise NotImplementedError
        self.model.to(self.device)

        if len(list(self.model.parameters())) != 0:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        else:
            self.optimizer = None

    def train_ensembling(self, seed):
        # assert isinstance(self.model, (SAdaGCN, AdaGCN, GBGCN))
        input_dict = self.get_input_dict(0)
        acc = self.model.train_and_test(input_dict)
        return acc

    def get_input_dict(self, epoch):
        if self.type_model in [
            "EnGCN",
        ]:
            input_dict = {
                "split_masks": self.split_masks,
                "data": self.data,
                "x": self.x,
                "y": self.y,
                "optimizer": self.optimizer,
                "loss_op": self.loss_op,
                "device": self.device,
            }
        else:
            Exception(f"the model of {self.type_model} has not been implemented")
        return input_dict

