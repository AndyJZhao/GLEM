"""GNNs using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch.nn as nn
import torch.nn.functional as F

MAX_N_LAYERS = 999

import dgl.nn.pytorch as dglnn


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation=F.relu, dropout=0.5, norm='BN', input_norm=True):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        norm_layer = nn.BatchNorm1d if norm == 'BN' else nn.LayerNorm

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            bias = i == n_layers - 1

            self.convs.append(dglnn.GraphConv(in_hidden, out_hidden, "both", bias=bias))
            if i < n_layers - 1:  # Not last layer (classification layer)
                self.norms.append(norm_layer(out_hidden))
        if input_norm:
            self.input_norm = norm_layer(in_feats)
        self.input_drop = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat, output_hidden_layer=None):
        if output_hidden_layer is None:  # Set to an impossible epoch number
            output_hidden_layer = MAX_N_LAYERS
        if hasattr(self, 'input_norm'):
            feat = self.input_norm(feat)
        h = feat
        h = self.input_drop(h)

        for i in range(self.n_layers):
            h = self.convs[i](graph, h)
            if i == output_hidden_layer:
                return h

            if i < self.n_layers - 1:
                h = self.norms[i](h)
                h = self.activation(h)
                h = self.dropout(h)

        return h
