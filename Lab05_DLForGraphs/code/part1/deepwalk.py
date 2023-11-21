"""
Deep Learning on Graphs - ALTEGRAD - Nov 2023

Ben Kabongo
M2 MVA
"""

import numpy as np
import networkx as nx
import random
from gensim.models import Word2Vec


############## Task 1
# Simulates a random walk of length "walk_length" starting from node "node"
def random_walk(G, node, walk_length):
    walk = [node]
    current_node = node
    for _ in range(1, walk_length):
        neighbors = list(G.neighbors(current_node))
        idx = random.randint(0, len(neighbors)-1)
        current_node = neighbors[idx]
        walk.append(current_node)
    walk = [str(node) for node in walk]
    random.shuffle(walk)
    return walk


############## Task 2
# Runs "num_walks" random walks from each node
def generate_walks(G, num_walks, walk_length):
    walks = []
    for node in G.nodes:
        walks.extend([random_walk(G, node, walk_length) for _ in range(num_walks)])
    return walks


# Simulates walks and uses the Skipgram model to learn node representations
def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(vector_size=n_dim, window=8, min_count=0, sg=1, workers=8, hs=1)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model


if __name__ == "__main__":
    BASE_PATH = "Labs/Lab05_DLForGraphs/code/"
    G = nx.read_weighted_edgelist(BASE_PATH + 'data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
    NUM_WALKS = 1
    WALK_LENGTH = 8
    N_DIM = 10
    model = deepwalk(G, NUM_WALKS, WALK_LENGTH, N_DIM)
    