"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2023

Ben Kabongo
M2 MVA
"""

import torch
import torch.nn as nn

class DeepSets(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(DeepSets, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        ############## Task 3
        if x.ndim == 1: x = x.unsqueeze(0)
        embedded = self.embedding(x)
        out = self.fc1(embedded)
        out = self.tanh(out)
        sum = out.sum(axis=1)
        y = self.fc2(sum)
        return y.squeeze()


class LSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        ############## Task 4
        if x.ndim == 1: x = x.unsqueeze(0)
        embedded = self.embedding(x)
        _, (hn, _) = self.lstm(embedded)
        y = self.fc(hn)
        return y.squeeze()
