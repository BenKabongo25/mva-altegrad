"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023

Ben Kabongo
M2 MVA
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score, adjusted_rand_score
from deepwalk import deepwalk


# Loads the karate network
BASE_PATH = "Labs/Lab_05_DLForGraphs/code/"
G = nx.read_weighted_edgelist(BASE_PATH + 'data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt(BASE_PATH + 'data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network

nx.draw_networkx(G, arrows=False, with_labels=True, labels=idx_to_class_label, 
                nodelist=list(idx_to_class_label.keys()),
                node_color=list(idx_to_class_label.values()),
                node_size=200, alpha=.5)
#plt.savefig("karate_visualisation.pdf")
#plt.show()


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Deep walk : Accuracy score Logistic Regression : {acc}") # 1.0


############## Task 8

def compute_normalized_laplacian(adjacency_matrix):
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    d_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix))
    normalized_laplacian = np.identity(adjacency_matrix.shape[0]) - np.dot(np.dot(d_inv_sqrt, adjacency_matrix), d_inv_sqrt)
    return normalized_laplacian

L = compute_normalized_laplacian(nx.to_numpy_array(G))
sp = SpectralEmbedding(n_components=2)
sp_embeddings = sp.fit_transform(L)

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = sp_embeddings[idx_train,:]
X_test = sp_embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Spectral embedding : Accuracy score Logistic Regression : {acc}") # 0.8571428571428571