"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023

Ben Kabongo
M2 MVA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    """Simple GNN model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, X, A):
        ############## Tasks 10 
        Z = self.dropout(self.relu(self.fc1(torch.mm(A, X))))
        Z = self.relu(self.fc2(torch.mm(A, Z)))
        Z = self.fc3(Z)
        return F.log_softmax(Z, dim=-1)

        ## for Task 13, see gnn_cora
