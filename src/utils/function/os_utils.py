import errno
import pickle
import os


def run_command_parallel(cmd, gpus, log_func=print):
    _ = cmd.split('python ')
    env_path, variables = _[0], _[1]
    cmd = f'CUDA_VISIBLE_DEVICES={gpus} {env_path}torchrun --master_port={find_free_port()} --nproc_per_node={len(gpus.split(","))} {variables}'
    run_command(cmd, log_func)


def run_command(cmd, log_func=print):
    log_func(f'Running command:\n{cmd}')
    ret_value = os.system(cmd)
    if ret_value != 0:
        raise ValueError(f'Failed to operate {cmd}')


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


def find_free_port():
    from contextlib import closing
    import socket
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def check_path_dict(path_dict):
    # Check if all paths in path_dict already exists.
    try:
        for k, p in path_dict.items():
            assert os.path.exists(p), f'{k} not found.'
        return True
    except:
        return False


def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file


def list_dir(dir_name, error_msg=None):
    try:
        f_list = os.listdir(dir_name)
        return f_list
    except FileNotFoundError:
        if error_msg is not None:
            print(f'{error_msg}')
        return []


def silent_remove(file_or_path):
    # Modified from 'https://stackoverflow.com/questions/10840533/most-pythonic-way-to-delete-a-file-which-may-not-exist'
    import shutil
    try:
        if file_or_path[-1] == '/':
            shutil.rmtree(file_or_path)
        else:
            os.remove(file_or_path)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def remove_file(f_list):
    'Remove file or file list'
    f_list = f_list if isinstance(f_list, list) else [f_list]
    for f_name in f_list:
        silent_remove(f_name)


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def get_grand_parent_dir(f_name):
    from pathlib import Path
    if '.' in f_name.split('/')[-1]:  # File
        return get_grand_parent_dir(get_dir_of_file(f_name))
    else:  # Path
        return f'{Path(f_name).parent}/'


def get_abs_path(f_name, style='command_line'):
    # python 中的文件目录对空格的处理为空格，命令行对空格的处理为'\ '所以命令行相关需 replace(' ','\ ')
    if style == 'python':
        cur_path = os.path.abspath(os.path.dirname(__file__))
    elif style == 'command_line':
        cur_path = os.path.abspath(os.path.dirname(__file__)).replace(' ', '\ ')

    root_path = cur_path.split('src')[0]
    return os.path.join(root_path, f_name)


def pickle_save(var, f_name):
    mkdir_list([f_name])
    pickle.dump(var, open(f_name, 'wb'))
    print(f'Saved {f_name}')


def pickle_load(f_name):
    return pickle.load(open(f_name, 'rb'))


import datetime
import json
import logging
# import os
import socket
import subprocess
import sys
import time

import numpy as np
import pytz

from utils.function.os_utils import init_path

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path.split('src')[0]


# *  <<<<<<<<<<<<<<<<<<<< GIT >>>>>>>>>>>>>>>>>>>>

def get_git_hash():
    return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], stdout=subprocess.PIPE).stdout.decode('utf-8').strip('\n')


# *  <<<<<<<<<<<<<<<<<<<< PROJ SHARED UTILS >>>>>>>>>>>>>>>>>>>>
def floor_quantize(val, to_values):
    """Quantize a value with regard to a set of allowed values.

    Examples:
        quantize(49.513, [0, 45, 90]) -> 45
        quantize(17, [0, 10, 20, 30]) -> 10 # FLOORED

    Note: function doesn't assume to_values to be sorted and
    iterates over all values (i.e. is rather slow).

    Args:
        val        The value to quantize
        to_values  The allowed values
    Returns:
        Closest value among allowed values.
    """
    best_match = None
    best_match_diff = None
    assert min(to_values) <= val
    for other_val in to_values:
        if other_val <= val:  # Floored (only smaller values are matched)
            diff = abs(other_val - val)
            if best_match is None or diff < best_match_diff:
                best_match = other_val
                best_match_diff = diff
    return best_match


def get_max_batch_size(gpu_mem, max_bsz_dict):
    quantized_gpu_mem = floor_quantize(gpu_mem, max_bsz_dict.keys())
    return max_bsz_dict[quantized_gpu_mem]


def calc_bsz_grad_acc(eq_batch_size, max_bsz_dict, sv_info, min_bsz=2):
    max_bsz_per_gpu = get_max_batch_size(sv_info.gpu_mem, max_bsz_dict)
    gpus = os.environ['CUDA_VISIBLE_DEVICES']
    n_gpus = len(gpus.split(',')) if gpus != '' else 1
    print(f'N-GPUs={n_gpus}')

    def find_grad_acc_steps(bsz_per_gpu):
        # Find batch_size and grad_acc_steps combination that are DIVISIBLE!
        grad_acc_steps = eq_batch_size / bsz_per_gpu / n_gpus
        if grad_acc_steps.is_integer():
            return bsz_per_gpu, int(grad_acc_steps)
        elif grad_acc_steps:
            if bsz_per_gpu >= min_bsz:
                return find_grad_acc_steps(bsz_per_gpu - 1)
            else:
                raise ValueError(f'Cannot find grad_acc_step with integer batch_size greater than {min_bsz}, eq_bsz={eq_batch_size}, n_gpus={n_gpus}')

    batch_size, grad_acc_steps = find_grad_acc_steps(max_bsz_per_gpu)
    print(f'Eq_batch_size = {eq_batch_size}, bsz={batch_size}, grad_acc_steps={grad_acc_steps}, ngpus={n_gpus}')
    return batch_size, grad_acc_steps


def json_save(data, file_name, log_func=print):
    with open(init_path(file_name), 'w', encoding='utf-8') as f:
        try:
            json.dumps(data)
        except:
            log_func(f"{data['Static logs']} failed to save in json format.")
        json.dump(data, f, ensure_ascii=False, indent=4)
        # log_func(f'Successfully saved to {file_name}')


def json_load(file_name):
    with open(file_name) as data_file:
        return json.load(data_file)


# * ============================= Init =============================

def exp_init(args):
    """
    Functions:
    - Set GPU
    - Initialize Seeds
    - Set log level
    """
    from warnings import simplefilter
    simplefilter(action='ignore', category=DeprecationWarning)
    # if not hasattr(args, 'local_rank'):
    if args.gpus is not None and args.gpus != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    if hasattr(args, 'local_rank') and args.local_rank > 1: block_log()
    # Torch related packages should be imported afterward setting
    init_random_state(args.seed)
    os.chdir(root_path)


def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
    import torch
    import random
    # import dgl
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret

    return wrapper


def is_runing_on_local():
    try:
        host_name = socket.gethostname()
        if 'MacBook' in host_name:
            return True
    except:
        print("Unable to get Hostname and IP")
    return False


# * ============================= Print Related =============================
def subset_dict(d, sub_keys):
    return {k: d[k] for k in sub_keys if k in d}


def print_dict(d, end_string='\n\n'):
    for key in d.keys():
        if isinstance(d[key], dict):
            print('\n', end='')
            print_dict(d[key], end_string='')
        elif isinstance(d[key], int):
            print('{}: {:04d}'.format(key, d[key]), end=', ')
        elif isinstance(d[key], float):
            print('{}: {:.4f}'.format(key, d[key]), end=', ')
        else:
            print('{}: {}'.format(key, d[key]), end=', ')
    print(end_string, end='')


def block_log():
    sys.stdout = open(os.devnull, 'w')
    logger = logging.getLogger()
    logger.disabled = True


def enable_logs():
    # Restore
    sys.stdout = sys.__stdout__
    logger = logging.getLogger()
    logger.disabled = False


def print_log(log_dict):
    log_ = lambda log: f'{log:.4f}' if isinstance(log, float) else f'{log:04d}'
    print(' | '.join([f'{k} {log_(v)}' for k, v in log_dict.items()]))


def mp_list_str(mp_list):
    return '_'.join(mp_list)


# * ============================= Time Related =============================

def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


# * ============================= Parser Related =============================
def parse_conf(parser, input):
    """Update parser by input (Dictionary or namespace)"""
    # Get default parser and update
    args = parser.parse_args([])
    d = input if type(input) == dict else input.__dict__
    args.__dict__.update({k: v for k, v in d.items() if k in args.__dict__})
    return args


def args_to_cmd(parser, input, allow_unknown_args=False, to_str=True):
    """Convert parser and input to args"""
    default = vars(parser.parse_args([]))
    d = input if type(input) == dict else input.__dict__
    type_spec_parse_func = {
        **{_: lambda k, v: f'--{k}={v}' for _ in (int, float, str)},
        bool: lambda k, v: f'--{k}' if default[k] != v else '',
        list: lambda k, v: f'--{k}={" ".join([str(_) for _ in v])}',
    }

    is_parse = lambda k: True if allow_unknown_args else lambda k: k in default
    parse_func = lambda k, v: type_spec_parse_func[type(v)](k, v) if is_parse(k) else ''
    rm_empty = lambda input_list: [_ for _ in input_list if len(_) > 0]
    cmd_list = rm_empty([parse_func(k, v) for k, v in d.items()])
    if to_str:
        return ' '.join(cmd_list)
    else:
        return cmd_list


# * ============================= Itertool Related =============================

def lot_to_tol(list_of_tuple):
    # list of tuple to tuple lists
    # Note: zip(* zipped_file) is an unzip operation
    return list(map(list, zip(*list_of_tuple)))
