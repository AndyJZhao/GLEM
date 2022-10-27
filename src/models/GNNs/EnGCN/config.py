from utils.data import SeqGraph
from utils.modules import ModelConfig
from utils.settings import *
from models.GNNs.gnn_utils import GNNConfig


class EnGCNConfig(GNNConfig):

    def __init__(self, args=None):
        super(EnGCNConfig, self).__init__(args)

        # No need to define any args since it's the simplest GCN and the args are defined in "GNNConfig"
        self._post_init(args)
