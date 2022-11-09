import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))
from utils import time_logger
from models.GLEM import GLEMConfig


@time_logger
def train_glem(args):
    # ! Init Arguments
    cf = GLEMConfig(args).init()
    # ! Import packages

    from models.GLEM.GLEM_trainer import GLEMTrainer
    GLEMTrainer(cf).glem_train()
    return


if __name__ == "__main__":
    args = GLEMConfig().parse_args()
    train_glem(args)
