from models.GNNs.gnn_utils import GNNConfig

class SAGEConfig(GNNConfig):

    def __init__(self, args=None):
        super(SAGEConfig, self).__init__(args)
        # ! GraphSAGE specific settings
        # E.g. different default lr, or fanouts
        self.model = 'SAGE'
        self.lr = 0.003
        self.dropout = 0.5
        self.n_layers = 3
        self.n_hidden = 256
        self.weight_decay = 0.0
        self.early_stop = 15
        self.epochs = 20

        self.fan_out = '5,10,15'
        self.num_workers = 0
        self.batch_size = 1000
        self.log_every = 20
        # ! Post Init Settings
        self._post_init(args)


    # *  <<<<<<<<<<<<<<<<<<<< PATH RELATED >>>>>>>>>>>>>>>>>>>>
    para_prefix = {**GNNConfig.para_prefix, 'fan_out': 'fan', 'batch_size': 'bs'}
    args_to_parse = list(para_prefix.keys()) + ['num_workers', 'log_every']