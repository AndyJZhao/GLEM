import argparse
import torch as th
import numpy as np
import os
from tqdm import tqdm
import gc
import pickle

def _judge_type(data):
    min_val, max_val = data.min(), data.max()
    _dtype = type(min_val)
    if np.issubdtype(_dtype, np.integer):
        if max_val <= 1 and min_val >= 0:
            _dtype = np._bool
        if max_val <= 255 and min_val >= 0:
            _dtype = np.uint8
        elif max_val <= 65535 and min_val >= 0:
            _dtype = np.uint16
        elif max_val <= 2147483647 and min_val >= -2147483647:
            _dtype = np.int32
    elif np.issubdtype(_dtype, np.float):
        _dtype = np.float16
    return _dtype

def save_memmap(data: np.ndarray, path, dtype=None, node_chunk_size=1000000, log=print):
    # ! Determine the least memory cost type

    dtype = _judge_type(data) if dtype is None else dtype

    # ! Store memory map
    x = np.memmap(path, dtype=dtype, mode='w+',
                  shape=data.shape)

    for i in tqdm(range(0, data.shape[0], node_chunk_size)):
        j = min(i + node_chunk_size, data.shape[0])
        x[i:j] = data[i:j]
    log(f'Saved {path} as {dtype}...')
    del x
    gc.collect()
    log('releas x')
    return # SN(type=dtype, path=path, shape=data.shape)

def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'

def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path): return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise

def mkdir_list(p_list, use_relative_path=True, log=True):
    """Create directories for the specified path lists.
        Parameters
        ----------
        p_list :Path lists or a single path

    """
    # ! Note that the paths MUST END WITH '/' !!!
    root_path = os.path.abspath(os.path.dirname(__file__)).split('src')[0]
    p_list = p_list if isinstance(p_list, list) else [p_list]
    for p in p_list:
        p = os.path.join(root_path, p) if use_relative_path else p
        p = get_dir_of_file(p)
        mkdir_p(p, log)

def pickle_save(var, f_name):
    mkdir_list([f_name])
    pickle.dump(var, open(f_name, 'wb'))
    print(f'Saved {f_name}')

def save_gnn_result(args, pred, res):
    save_memmap(pred.cpu().numpy(), init_path(os.path.join(args.out_put,'.pred')), np.float16)
    pickle_save(res, os.path.join(args.out_put,'.result'))

def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file

def main(args):
    val_acc = args.val_acc
    test_acc = args.test_acc
    res = {'val_acc': val_acc, 'test_acc': test_acc}
    if args.pred_path:
        preds = th.load(args.pred_path)
    else:
        preds = None
        print('No preds')

    save_gnn_result(args, preds, res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GMLP")
    parser.add_argument("--val_acc", type=float, default=None)
    parser.add_argument("--test_acc", type=float, default=None)
    parser.add_argument("--pred_path", type=str, default='/home/v-haoyan1/')
    parser.add_argument("--out_put", type=str, default='/home/v-haoyan1/')

    args = parser.parse_args()
    print(args)
    main(args)