"""
Graph Mining - ALTEGRAD - Oct 2023

Ben Kabongo B.
MVA, ENS Paris-Saclay
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


############## Task 10
# Generate simple dataset
def create_dataset():
    Gs = list()
    y = list()

    for n in range(3, 103):
        Gs.append(nx.cycle_graph(n))
        y.append(0)
        Gs.append(nx.path_graph(n))
        y.append(1)

    return Gs, y


Gs, y = create_dataset()
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.1)

# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):    
    all_paths = dict()
    sp_counts_train = dict()
    
    for i,G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)
       
    sp_counts_test = dict()

    for i,G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)


    phi_train = np.zeros((len(Gs_train), len(all_paths)))
    for i in range(len(Gs_train)):
        for length in sp_counts_train[i]:
            phi_train[i,all_paths[length]] = sp_counts_train[i][length]
  
    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i,all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test



############## Task 11
# Compute the graphlet kernel
def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    
    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0,1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0,1)
    graphlets[2].add_edge(1,2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0,1)
    graphlets[3].add_edge(1,2)
    graphlets[3].add_edge(0,2)

    
    phi_train = np.zeros((len(Gs_train), 4))
    for i, G in enumerate(Gs_train):
        for _ in range(n_samples):
            nodes = np.random.choice(G.nodes(), size=3, replace=False)
            subgraph = G.subgraph(nodes)
            for g in range(4):
                phi_train[i, g] += int(nx.is_isomorphic(subgraph, graphlets[g]))

    phi_test = np.zeros((len(Gs_test), 4))
    for i, G in enumerate(G_test):
        for _ in range(n_samples):
            nodes = np.random.choice(G.nodes(), size=3, replace=False)
            subgraph = G.subgraph(nodes)
            for g in range(4):
                phi_train[i, g] += int(nx.is_isomorphic(subgraph, graphlets[g]))


    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)

# Task 12
K_train_gk, K_test_gk = graphlet_kernel(G_train, G_test)

# Task 13
# Initialize SVM and train
for i, (K_train, K_test) in enumerate([(K_train_sp, K_test_sp), (K_train_gk, K_test_gk)]):
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, y_train)
    y_pred = clf.predict(K_test)
    acc = accuracy_score(y_test, y_pred)
    print(["Shortest path", "Graphlet"][i] + " kernel accuracy", acc)
print()


### 
#for n_samples in range(50, 501, 50):
#    K_train, K_test = graphlet_kernel(G_train, G_test, n_samples)
#    clf = SVC(kernel="precomputed") 
#    clf.fit(K_train, y_train)
#    y_pred = clf.predict(K_test)
#    acc = accuracy_score(y_test, y_pred)
#    print(f"Graphlet kernel n_samples={n_samples} accuracy : {acc}")

### Question 6
#P4, S4 = nx.Graph(), nx.Graph()
#P4.add_nodes_from(list("ABCD"))
#S4.add_nodes_from(range(1,5))

#P4.add_edge("A","B")
#P4.add_edge("B","C")
#P4.add_edge("C","D")

#S4.add_edge(1,2)
#S4.add_edge(1,3)
#S4.add_edge(1,4)

#shortest_path_kernel([P4, P4], [P4, S4])
