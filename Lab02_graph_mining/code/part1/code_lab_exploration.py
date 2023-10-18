"""
Graph Mining - ALTEGRAD - Oct 2023

Ben Kabongo B.
MVA, ENS Paris-Saclay
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1
BASE_PATH = "Labs/Lab02_graph_mining/code"

G = nx.read_edgelist(
    path=BASE_PATH + "/datasets/CA-HepTh.txt",
    comments="#",
    delimiter="\t"
)

g_number_nodes = G.number_of_nodes()
g_number_edges = G.number_of_edges()
print("Number of nodes :", g_number_nodes)
print("Number of edges :", g_number_edges)
print()


############## Task 2

connected_components = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
print("Number of connected components :", len(connected_components))
if len(connected_components) > 1:
    largest_cc = max(nx.connected_components(G), key=len)
    CG = G.subgraph(largest_cc).copy()
    sg_number_nodes = CG.number_of_nodes()
    sg_number_edges = CG.number_of_edges()
    print("The graph is not connected !")
    print("Number of nodes of the largest connected component :", sg_number_nodes)
    print("That represent %s/100 of the number total of nodes in the graph" % round((sg_number_nodes * 100 / g_number_nodes), 2))
    print("Number of edges of the largest connected component :", sg_number_edges)
    print("That represent %s/100 of the number total of edges in the graph" % round((sg_number_edges * 100 / g_number_edges), 2))
    print()


############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]
min_d = np.min(degree_sequence)
max_d = np.max(degree_sequence)
mean_d = np.mean(degree_sequence)
median_d = np.median(degree_sequence)
std_d = np.std(degree_sequence)
print("Degree statistics")
print("Min \t:", min_d)
print("Max \t:", max_d)
print("Mean \t:", mean_d)
print("Median \t:", median_d)
print("Std \t:", std_d)
print()


############## Task 4
degree_histogram = nx.degree_histogram(G)
plt.figure()
plt.title("Degree histogram")
plt.hist(degree_histogram)
#plt.show()
plt.savefig(BASE_PATH + "/degree_histogram.png")

plt.figure()
plt.title("Degree histogram - Log-log axis")
plt.loglog(degree_histogram)
#plt.show()
plt.savefig(BASE_PATH + "/degree_histogram_loglog.png")


############## Task 5
print("Global clustering coefficient :", nx.transitivity(G))
print()

### Question 3
#for n in range(3, 10):
#    g = nx.Graph()
#    g.add_nodes_from(range(n))
#    for i in range(n-1):
#        g.add_edge(i, i+1)
#    g.add_edge(n-1, 0)
#    print(n,":", nx.transitivity(g))