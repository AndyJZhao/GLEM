import dgl
import numpy as np
import torch as th
from utils.function import init_random_state


def sample_nodes(g, seed_nodes, fanout_list):
    # seed_nodes = th.tensor(seed_nodes).to(g.device) if isinstance(seed_nodes, int) else seed_nodes
    induced_nodes = {0: (cur_nodes := seed_nodes.view(-1))}
    init_random_state(0)
    for l, fanout in enumerate(fanout_list):
        frontier = dgl.sampling.sample_neighbors(g, cur_nodes, fanout)
        cur_nodes = frontier.edges()[0].unique()
        induced_nodes[l + 1] = cur_nodes
    sampled_nodes = th.cat(list(induced_nodes.values())).unique()
    return sampled_nodes, induced_nodes


def get_edge_set(g: dgl.DGLGraph):
    """graph_edge_to list of (row_id, col_id) tuple
    """

    return set(map(tuple, np.column_stack([_.cpu().numpy() for _ in g.edges()]).tolist()))


def edge_set_to_inds(edge_set):
    """ Unpack edge set to row_ids, col_ids"""
    return list(map(list, zip(*edge_set)))
