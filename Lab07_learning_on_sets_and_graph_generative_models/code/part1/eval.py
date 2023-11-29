"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2023

Ben Kabongo
M2 MVA
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error
import torch

from utils import create_test_dataset
from models import DeepSets, LSTM

# Initializes device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
batch_size = 64
embedding_dim = 128
hidden_dim = 64

# Generates test data
X_test, y_test = create_test_dataset()
cards = [X_test[i].shape[1] for i in range(len(X_test))]
n_samples_per_card = X_test[0].shape[0]
n_digits = 11

# Retrieves DeepSets model
deepsets = DeepSets(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading DeepSets checkpoint!")
checkpoint = torch.load('model_deepsets.pth.tar')
deepsets.load_state_dict(checkpoint['state_dict'])
deepsets.eval()

# Retrieves LSTM model
lstm = LSTM(n_digits, embedding_dim, hidden_dim).to(device)
print("Loading LSTM checkpoint!")
checkpoint = torch.load('model_lstm.pth.tar')
lstm.load_state_dict(checkpoint['state_dict'])
lstm.eval()

# Dict to store the results
results = {'deepsets': {'acc':[], 'mae':[]}, 'lstm': {'acc':[], 'mae':[]}}

for i in range(len(cards)):
    y_pred_deepsets = list()
    y_pred_lstm = list()

    for j in range(0, n_samples_per_card, batch_size): 
        ############## Task 6
        x_batch = X_test[i][j : min(n_samples_per_card, j + batch_size)]
        x_batch = torch.LongTensor(x_batch)

        y_pred_deepsets_batch = deepsets(x_batch)
        y_pred_lstm_batch = lstm(x_batch)

        y_pred_deepsets.append(y_pred_deepsets_batch)
        y_pred_lstm.append(y_pred_lstm_batch)

    y_pred_deepsets = torch.cat(y_pred_deepsets)
    y_pred_deepsets = y_pred_deepsets.detach().cpu().numpy()
    
    acc_deepsets = accuracy_score(np.floor(y_pred_deepsets), y_test[i])
    mae_deepsets = mean_absolute_error(y_test[i], y_pred_deepsets)
    results['deepsets']['acc'].append(acc_deepsets)
    results['deepsets']['mae'].append(mae_deepsets)
    
    y_pred_lstm = torch.cat(y_pred_lstm)
    y_pred_lstm = y_pred_lstm.detach().cpu().numpy()
    y_pred_lstm = np.floor(y_pred_lstm)
    
    acc_lstm = accuracy_score(np.floor(y_pred_lstm), y_test[i])
    mae_lstm = mean_absolute_error(y_test[i], y_pred_lstm)
    results['lstm']['acc'].append(acc_lstm)
    results['lstm']['mae'].append(mae_lstm)


############## Task 7
    
plt.style.use("bmh")

plt.figure()

plt.title("Accuracies")
plt.plot(cards, results['deepsets']['acc'], label="DeepSet")
plt.plot(cards, results['lstm']['acc'], label="LSTM")
plt.xticks(cards)
plt.legend()
plt.savefig("acc.png")
plt.show()

plt.figure()
plt.title("Mean absolute accuracies")
plt.plot(cards, results['deepsets']['mae'], label="DeepSet")
plt.plot(cards, results['lstm']['mae'], label="LSTM")
plt.xticks(cards)
plt.legend()
plt.savefig("mae.png")
plt.show()