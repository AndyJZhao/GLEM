import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))
from utils import time_logger
from models.GLEM import GLEMConfig


@time_logger
def graph_vf_training(args):
    # ! Init Arguments
    cf = GLEMConfig(args).init()
    # ! Import packages
    # Note that the assignment of GPU-ID must be specified before torch/dgl is imported.
    from models.GLEM.GLEM_trainer import GLEMTrainer
    GLEMTrainer(cf).glem_train()
    return cf


if __name__ == "__main__":
    args = GLEMConfig().parse_args()
    graph_vf_training(args)
