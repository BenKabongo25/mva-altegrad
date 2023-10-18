"""
Graph Mining - ALTEGRAD - Oct 2023

Ben Kabongo B.
MVA, ENS Paris-Saclay
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans


############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    print("Graph to array ...")
    A = nx.to_numpy_array(G)
    n = A.shape[0]
    D = np.diag(A.sum(1).flatten())
    L = np.identity(n) - np.linalg.inv(D) @ A
    d = k
    print("Eig values ...")
    _, U = eigs(L, k=d, which="SR")
    kmeans = KMeans(n_clusters=k)
    print("KMeans ...")
    kmeans.fit(np.real(U))
    labels = kmeans.labels_
    print("Clustering ...")
    clustering = {}
    nodes = list(G.nodes())
    for i in range(n):
        clustering[nodes[i]] = labels[i]
    return clustering


############## Task 7
BASE_PATH = "Labs/Lab02_graph_mining/code"

print("Graph loading ...")
G = nx.read_edgelist(
    path=BASE_PATH + "/datasets/CA-HepTh.txt",
    comments="#",
    delimiter="\t"
)
print("Giant connected component ...")
largest_cc = max(nx.connected_components(G), key=len)
CG = G.subgraph(largest_cc).copy()
k = 50

print("Clustering ...")
clustering = spectral_clustering(CG, k)
#print(clustering)


############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    m = G.number_of_edges()
    Q = 0
    clusters = np.unique(list(clustering.values()))
    for c in clusters:
        nodes = [node for node, cluster in clustering.items() if cluster==c]
        c_graph = G.subgraph(nodes)
        lc = c_graph.number_of_edges()
        dc = sum(dict(c_graph.degree()).values())
        qc = (lc/m) - ((.5*dc/m) ** 2)
        Q += qc
    return Q


############## Task 9
random_clustering = {node: randint(1, k) for node in CG.nodes()}

print("Modularity of spectral clustering :", modularity(CG, clustering))
print("Modularity of random partition :", modularity(CG, random_clustering)) 
