"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023

Ben Kabongo
M2 MVA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, n_class):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, n_class)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj, idx):
        ############## Task 2
        A = adj + torch.eye(adj.size(0))

        z0 = self.relu( torch.mm(A, self.fc1(x_in)) )
        z1 = self.relu( torch.mm(A, self.fc2(z0)) )

        idx = idx.unsqueeze(1).repeat(1, z1.size(1))
        zg = torch.zeros(torch.max(idx).int().item()+1, z1.size(1), device=x_in.device)
        zg = zg.scatter_add_(0, idx, z1) 

        y = self.fc4(self.relu(self.fc3(zg)))
        return F.log_softmax(y, dim=1)
