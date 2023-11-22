"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023

Ben Kabongo
M2 MVA
"""

import networkx as nx
import numpy as np
import torch
from random import randint

def create_dataset():
    Gs = list()
    y = list()

    ############## Task 1
    for n in range(10, 20, 1):
        Gs.extend([nx.fast_gnp_random_graph(n=n, p=0.2) for _ in range(5)])
        y.extend([0] * 5)
        Gs.extend([nx.fast_gnp_random_graph(n=n, p=0.4) for _ in range(5)])
        y.extend([1] * 5)
    return Gs, y


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

