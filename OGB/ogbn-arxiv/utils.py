from texttable import Texttable
from torch_sparse import SparseTensor
import torch
import numpy as np

MB = 1024 ** 2
GB = 1024 ** 3

def print_args(args):
    _dict = vars(args)
    t = Texttable()
    t.add_row(["Parameter", "Value"])
    for k in _dict:
        # if k in ['lr', 'dst_sample_rate', 'dst_walk_length', 'dst_update_interval', 'dst_update_rate']:
        t.add_row([k, _dict[k]])
    print(t.draw())
