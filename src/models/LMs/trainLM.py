import os.path as osp
import sys

sys.path.append((osp.abspath(osp.dirname(__file__)).split('src')[0] + 'src'))
from models.LMs.lm_utils import *

if __name__ == "__main__":
    # ! Init Arguments
    model = get_lm_model()
    Config, Trainer = get_lm_config(model), get_lm_trainer(model)
    args = Config().parse_args()
    cf = Config(args).init()

    # ! Load data and train
    trainer = Trainer(cf=cf)
    trainer.train()
    trainer.eval_and_save()
