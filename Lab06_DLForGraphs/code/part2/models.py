"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023

Ben Kabongo
M2 MVA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MessagePassing(nn.Module):
    def __init__(self, input_dim, output_dim, neighbor_aggr):
        super(MessagePassing, self).__init__()
        self.neighbor_aggr = neighbor_aggr
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, X, A):
        ############## Task 6
        Z1 = self.fc1(X)
        Z2 = self.fc2(X)
        if self.neighbor_aggr == 'sum':
            out = Z1 + torch.mm(A, Z2)
        elif self.neighbor_aggr == 'mean':
            D = torch.diag(A.sum(0))
            out = Z1 + torch.mm( torch.mm(torch.inverse(D), A), Z2 )
        return out



class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, neighbor_aggr, readout, dropout):
        super(GNN, self).__init__()
        self.readout = readout
        self.mp1 = MessagePassing(input_dim, hidden_dim, neighbor_aggr)
        self.mp2 = MessagePassing(hidden_dim, hidden_dim, neighbor_aggr)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, X, A, idx):
        ############## Task 7
        Z0 = self.dropout(self.relu(self.mp1(X, A)))
        Z1 = self.relu(self.mp2(Z0, A))

        idx = idx.unsqueeze(1).repeat(1, Z1.size(1))
        ZG0 = torch.zeros(torch.max(idx).int().item()+1, Z1.size(1), device=Z1.device)
        if self.readout == 'sum':
            ZG0 = ZG0.scatter_add_(0, idx, Z1) 
        elif self.readout == 'mean':
            ZG0 = ZG0.scatter_add_(0, idx, Z1)
            count = torch.zeros(torch.max(idx).int().item()+1, Z1.size(1), device=Z1.device)
            count = count.scatter_add_(0, idx, torch.ones_like(Z1, device=Z1.device))
            ZG0 = torch.div(ZG0, count)

        ZG1 = self.fc(ZG0)
        
        return ZG1