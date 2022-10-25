import json
import os
import time

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch_geometric.transforms import SIGN
from torch_sparse import SparseTensor

from .utils import GB, MB, compute_tensor_bytes, get_memory_usage


class PrecomputingBase(torch.nn.Module):
    def __init__(self, args, data, train_idx, processed_dir):
        super(PrecomputingBase, self).__init__()

        self.num_layers = args.num_layers
        self.dim_hidden = args.dim_hidden
        self.num_classes = args.num_classes
        self.dataset = args.dataset
        self.type_model = args.type_model
        self.interval = args.eval_steps
        self.num_feats = args.num_feats
        self.batch_size = args.batch_size
        self.dropout = args.dropout
        self.norm = args.norm
        self.epochs = args.epochs
        self.multi_label = args.multi_label
        self.debug_mem_speed = args.debug_mem_speed
        self.processed_dir = processed_dir
        self.precompute(data, self.processed_dir)
        self.saved_args = vars(args)

    def precompute(self, data, processed_dir):
        # try:
        #     t1 = time.time()
        #     self.xs = torch.load(os.path.join(processed_dir, 'pre_xs.pt'))
        #     t2 = time.time()
        #     print("cached features are loaded using %.4f ." % (t2-t1))
        # except Exception as e:
        #     print("precomputing features, may take a while.")
        #     t1 = time.time()
        #     data = SIGN(self.num_layers)(data)
        #     self.xs = [data.x] + [data[f"x{i}"] for i in range(1, self.num_layers + 1)]
        #     t2 = time.time()
        #     print("precomputing finished using %.4f ." % (t2-t1))
        #     torch.save(self.xs, os.path.join(processed_dir, "pre_xs.pt"))
        #     print("precomputed features are cached.")
        print("precomputing features, may take a while.")
        t1 = time.time()
        data = SIGN(self.num_layers)(data)
        self.xs = [data.x] + [data[f"x{i}"] for i in range(1, self.num_layers + 1)]
        t2 = time.time()
        print("precomputing finished using %.4f s." % (t2 - t1))

    def forward(self, xs):
        raise NotImplementedError

    def mem_speed_bench(self, input_dict):
        split_idx = input_dict["split_masks"]
        device = input_dict["device"]
        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]
        torch.cuda.empty_cache()
        model_opt_usage = get_memory_usage(0, False)
        usage_dict = {
            "model_opt_usage": model_opt_usage,
            "data_mem": [],
            "act_mem": [],
            "peak_mem": [],
            "duration": [],
        }
        print(
            "model + optimizer only, mem: %.2f MB"
            % (usage_dict["model_opt_usage"] / MB)
        )
        xs_train = torch.cat([x[split_idx["train"]] for x in self.xs], -1)
        y_train = input_dict["y"][split_idx["train"]]
        dim_feat = self.xs[0].shape[-1]
        train_set = torch.utils.data.TensorDataset(xs_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, num_workers=8, pin_memory=True
        )
        epoch_start_time = time.time()
        torch.cuda.synchronize()
        for xs, y in train_loader:
            iter_start_time = time.time()
            torch.cuda.synchronize()
            xs = [x.to(device) for x in torch.split(xs, dim_feat, -1)]
            y = y.to(device)
            init_mem = get_memory_usage(0, False)
            data_mem = init_mem - usage_dict["model_opt_usage"]
            usage_dict["data_mem"].append(data_mem)
            print("data mem: %.2f MB" % (data_mem / MB))
            optimizer.zero_grad()
            out = self.forward(xs)
            if isinstance(loss_op, torch.nn.NLLLoss):
                out = F.log_softmax(out, dim=-1)
            elif isinstance(loss_op, torch.nn.BCEWithLogitsLoss):
                y = y.float()
            loss = loss_op(out, y)
            loss = loss.mean()
            before_backward = get_memory_usage(0, False)
            act_mem = before_backward - init_mem - compute_tensor_bytes([loss, out])
            usage_dict["act_mem"].append(act_mem)
            print("act mem: %.2f MB" % (act_mem / MB))
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            iter_end_time = time.time()
            duration = iter_end_time - iter_start_time
            print("duration: %.4f sec" % duration)
            usage_dict["duration"].append(duration)
            peak_usage = torch.cuda.max_memory_allocated(0)
            usage_dict["peak_mem"].append(peak_usage)
            print(f"peak mem usage: {peak_usage / MB}")
        with open(
            "./%s_%s_mem_speed_log.json"
            % (self.saved_args["dataset"], self.__class__.__name__),
            "w",
        ) as fp:
            info_dict = {**self.saved_args, **usage_dict}
            del info_dict["device"]
            json.dump(info_dict, fp)
        exit()

    def train_net(self, input_dict):
        split_idx = input_dict["split_masks"]
        device = input_dict["device"]
        optimizer = input_dict["optimizer"]
        loss_op = input_dict["loss_op"]

        xs_train = torch.cat([x[split_idx["train"]] for x in self.xs], -1)
        y_train = input_dict["y"][split_idx["train"]]
        dim_feat = self.xs[0].shape[-1]

        train_set = torch.utils.data.TensorDataset(xs_train, y_train)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, num_workers=8, pin_memory=True
        )

        total_correct = 0
        y_true, y_preds = [], []

        for xs, y in train_loader:
            xs = [x.to(device) for x in torch.split(xs, dim_feat, -1)]
            y = y.to(device)
            optimizer.zero_grad()
            out = self.forward(xs)
            if isinstance(loss_op, torch.nn.NLLLoss):
                out = F.log_softmax(out, dim=-1)
            elif isinstance(loss_op, torch.nn.BCEWithLogitsLoss):
                y = y.float()
            loss = loss_op(out, y)

            loss = loss.mean()
            loss.backward()
            optimizer.step()

            if not self.multi_label:
                y_preds.append(out.argmax(dim=-1).detach().cpu())
                y_true.append(y.detach().cpu())
                # total_correct += int(out.argmax(dim=-1).eq(y).sum())
            else:
                y_preds.append(out.detach().cpu())
                y_true.append(y.detach().cpu())
                # train_acc = f1_score(y_true[self.split_masks['train']],
                #                     pred[self.split_masks['train']], average='micro') \
                # if pred[self.split_masks['train']].sum() > 0 else 0
                # total_correct += int(out.eq(y).sum())

        y_true = torch.cat(y_true, 0)
        y_preds = torch.cat(y_preds, 0)
        if not self.multi_label:
            total_correct = y_preds.eq(y_true).sum().item()
            train_acc = float(total_correct / y_train.size(0))
        else:
            y_preds = (y_preds > 0).float().numpy()
            train_acc = f1_score(y_true, y_preds, average="micro")

        return float(loss.item()), train_acc

    @torch.no_grad()
    def inference(self, input_dict):
        x_all = input_dict["x"]
        device = input_dict["device"]
        y_preds = []
        loader = DataLoader(range(x_all.size(0)), batch_size=100000)
        for perm in loader:
            y_pred = self.forward([x[perm].to(device) for x in self.xs])
            y_preds.append(y_pred.cpu())
        y_preds = torch.cat(y_preds, dim=0)

        return y_preds