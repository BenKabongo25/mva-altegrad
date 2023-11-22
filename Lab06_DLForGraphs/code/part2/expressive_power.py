"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023

Ben Kabongo
M2 MVA
"""

import itertools
import networkx as nx
import numpy as np
import scipy.linalg
import scipy.sparse as sp
import torch

from models import GNN
from utils import sparse_mx_to_torch_sparse_tensor

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
input_dim = 1
hidden_dim = 32
output_dim = 4
dropout = 0.0
neighbor_aggr = 'mean'
readout = 'mean'


############## Task 4      
Gs = [nx.cycle_graph(n) for n in range(10, 20)]


############## Task 5
adj_batch = list()
idx_batch = list()

N = 0
for gi, G in enumerate(Gs):
    adj = nx.to_numpy_array(G)
    n = adj.shape[0]
    N += n
    adj_batch.append(adj)
    idx_batch.extend([gi] * n)


adj_batch = scipy.linalg.block_diag(*adj_batch)
# adj_batch = sp.csr_matrix(adj_batch)  # humm ... utilité ??
# adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch)

adj_batch = torch.Tensor(adj_batch)
features_batch = torch.ones((N, input_dim))
idx_batch = torch.LongTensor(idx_batch)


############## Task 8
def task8():
    print("[Task 8]")
    for neighbor_aggr, readout in [('mean', 'mean'), ('mean', 'sum'), ('sum', 'mean'), ('sum', 'sum')]:
        model = GNN(input_dim, hidden_dim, output_dim, neighbor_aggr, readout, dropout)
        embeddings = model(features_batch, adj_batch, idx_batch)

        print(f"Configuration : neighbor_aggr={neighbor_aggr}, readout={readout}")
        print(embeddings)
        print("==================================================================\n")
task8()


############## Task 9
G1, G2 = nx.union(nx.cycle_graph(3), nx.cycle_graph(3), rename=('L-', 'R-')), nx.cycle_graph(6)

############## Task 10
Gs = [G1, G2]
adj_batch = list()
idx_batch = list()

N = 0
for gi, G in enumerate(Gs):
    adj = nx.to_numpy_array(G)
    n = adj.shape[0]
    N += n
    adj_batch.append(adj)
    idx_batch.extend([gi] * n)


adj_batch = scipy.linalg.block_diag(*adj_batch)
# adj_batch = sp.csr_matrix(adj_batch)  # humm ... utilité ??
# adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch)

adj_batch = torch.Tensor(adj_batch)
features_batch = torch.ones((N, input_dim))
idx_batch = torch.LongTensor(idx_batch)

############## Task 11
model = GNN(input_dim, hidden_dim, output_dim, neighbor_aggr="sum", readout="sum", dropout=dropout)
embeddings = model(features_batch, adj_batch, idx_batch)
print("[Task 11]")
print(embeddings)
