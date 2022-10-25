import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))
from utils import time_logger
from models.GraphVF import GraphVFConfig


@time_logger
def graph_vf_training(args):
    # ! Init Arguments
    cf = GraphVFConfig(args).init()
    # ! Import packages
    # Note that the assignment of GPU-ID must be specified before torch/dgl is imported.
    from models.GraphVF.gvf_trainer import GVFTrainer
    GVFTrainer(cf).gvf_train()
    return cf


if __name__ == "__main__":
    args = GraphVFConfig().parse_args()
    graph_vf_training(args)
