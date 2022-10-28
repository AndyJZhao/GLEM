from utils.data import SeqGraph
from utils.modules import ModelConfig
from utils.settings import *
from models.GNNs.gnn_utils import GNNConfig


class EnGCNConfig(GNNConfig):

    def __init__(self, args=None):
        super(EnGCNConfig, self).__init__(args)
        # This will be completed in future
        self.model = 'EnGCN'



        # ! Post Init Settings
        self._post_init(args)

    # *  <<<<<<<<<<<<<<<<<<<< PATH RELATED >>>>>>>>>>>>>>>>>>>>
