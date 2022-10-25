from models.LMs.lm_utils import *


class BertConfig(LMConfig):

    def __init__(self, args=None):
        super(BertConfig, self).__init__(args)
        self.model = 'Bert'
        self._post_init(args)

    para_prefix = {**LMConfig.para_prefix}
    args_to_parse = list(para_prefix.keys())
    meta_data = {
        'Bert':
            SN(
                hf_model='bert-base-uncased',
                hidden_dim=768,
                max_bsz=SN(  # Batch size for different device
                    train={12: 8, 16: 12, 24: 16, 32: 24},
                    inf={12: 150, 16: 200, 24: 300, 32: 360},
                ),
                prt_lm={  # Initial LM configs
                    'arxiv': SN(
                        model='FtV1',
                        cmd='--lr=2e-05 --eq_batch_size=48 --weight_decay=0.01 --dropout=0.1 --att_dropout=0.1 --cla_dropout=0.1 --cla_bias=T --epochs=4 --warmup_epochs=0.2 --eval_patience=30482',
                        max_n_gpus=4, ),
                    'products': SN(
                        model='FtV1',
                        cmd='--lr=2e-05 --eq_batch_size=192 --weight_decay=0.01 --dropout=0.1 --att_dropout=0.3 --cla_dropout=0.2 --cla_bias=T --warmup_epochs=0.2 --eval_patience=65308 --epochs=4 --label_smoothing_factor=0.1 --warmup_epochs=0.6',
                        max_n_gpus=8,
                    )
                },
            ),
        'TinyBert':
            SN(
                hf_model='prajjwal1/bert-tiny',
                hidden_dim=128,
                max_bsz=SN(  # Batch size for different device
                    train={12: 8, 16: 18, 24: 12, 32: 72},
                    inf={12: 150, 16: 200, 24: 10, 32: 400},
                ),
                prt_lm={  # Initial LM configs
                    'arxiv': SN(
                        model='FtV1',
                        cmd='--att_dropout=0.1 --cla_dropout=0.4 --dropout=0.3 --epochs=4 --eq_batch_size=36 --eval_patience=50000 --label_smoothing_factor=0.3 --load_best_model_at_end=T --lr=2e-05 --warmup_epochs=0.6',
                        max_n_gpus=4,
                    ),
                    'products': SN(
                        model='FtV1',
                        cmd='--lr=2e-05 --eq_batch_size=144 --weight_decay=0.01 --dropout=0.1 --att_dropout=0.1 --cla_dropout=0.1 --cla_bias=T --epochs=2 --warmup_epochs=0.2 --eval_patience=50000',
                        max_n_gpus=8,
                    )
                },
            ),
    }
