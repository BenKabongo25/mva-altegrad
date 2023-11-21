"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023

Ben Kabongo
M2 MVA
"""

import networkx as nx
import numpy as np

def relu(x):
    return np.maximum(0, x)

C4, S4 = nx.cycle_graph(4), nx.Graph()
S4.add_nodes_from(range(1,5))
S4.add_edge(1,2)
S4.add_edge(1,3)
S4.add_edge(1,4)


for name_G, G in zip(["Cycle graph", "Star graph"], [C4, S4]):
    print("============================================================")
    print(name_G)

    A = nx.to_numpy_array(G)
    D = np.diag(np.sum(A, axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    I = np.identity(A.shape[0])
    A_normalized = np.dot(np.dot(D_inv_sqrt, A + I), D_inv_sqrt)
    
    print()
    print("A")
    print(A)

    print()
    print("Degree matrix")
    print(D)

    print()
    print("D inv sqrt")
    print(D_inv_sqrt)

    print()
    print("A normalizd")
    print(A_normalized)

    A = A_normalized
    X = np.ones((4, 1))
    W0 = np.array([[0.5, 0.2]])
    W1 = np.array([[0.3, -0.4, 0.8, 0.5], [-1.1, 0.6, -0.1, 0.7]])

    AX = np.dot(A, X)
    print()
    print("AX")
    print(AX)

    AXW0 = np.dot(AX, W0)
    print()
    print("AXW0")
    print(AXW0)

    Z0 = relu(AXW0)
    print()
    print("Z0")
    print(Z0)

    AZ0 = np.dot(A, Z0)
    print()
    print("AZ0")
    print(AZ0)

    AZ0W1 = np.dot(AZ0, W1)
    print()
    print("AZ0W1")
    print(AZ0W1)

    Z1 = relu(AZ0W1)
    print()
    print("Z1")
    print(Z1)
    



    

