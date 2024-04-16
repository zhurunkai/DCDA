import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def normt_spm(mx, method="in"):
    if method == "in":
        mx = mx.transpose()
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx
    if method == "sym":
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.0
        r_mat_inv = sp.diags(r_inv)
        mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
        return mx


def spm_to_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None
        self.layer = nn.Linear(in_channels, out_channels)

        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = None

    def forward(self, inputs, adj):
        if self.dropout is not None:
            inputs = self.dropout(inputs)
        w_input = torch.mm(inputs, self.layer.weight.T)
        adj = adj.to_sparse_csr()
        outputs = torch.mm(adj.to(self.layer.weight.device), w_input) + self.layer.bias

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs.type(torch.float32)


class GCN(nn.Module):
    def __init__(self, adj, in_channels, out_channels, hidden_layers):
        super().__init__()
        self.set_adj(adj)
        hl = hidden_layers.split(",")
        if hl[-1] == "d":
            dropout_last = True
            hl = hl[:-1]
        else:
            dropout_last = False

        i = 0
        layers = []
        last_c = in_channels
        for c in hl:
            if c[0] == "d":
                dropout = True
                c = c[1:]
            else:
                dropout = False
            c = int(c)

            i += 1
            conv = GraphConv(last_c, c, dropout=dropout)
            self.add_module("conv{}".format(i), conv)
            layers.append(conv)

            last_c = c

        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last)
        self.add_module("conv-last", conv)
        layers.append(conv)

        self.layers = layers

    def set_adj(self, adj):
        adj = normt_spm(adj, method="in")
        adj = spm_to_tensor(adj)
        self.adj = adj

    def forward(self, x):
        for conv in self.layers:
            x = conv(x, self.adj)
        return F.normalize(x)
