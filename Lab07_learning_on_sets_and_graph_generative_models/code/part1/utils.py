"""
Learning on Sets and Graph Generative Models - ALTEGRAD - Nov 2023

Ben Kabongo
M2 MVA
"""

import numpy as np


def create_train_dataset(n_train=100_000, max_train_card=10):
    ############## Task 1
    X_train = np.zeros((n_train, max_train_card))
    y_train = np.zeros(n_train)
    for i in range(n_train):
        ni = np.random.randint(1, 11)
        xi = np.random.choice(np.arange(1, 11), size=ni, replace=False)
        yi = xi.sum()
        xi = np.concatenate(([0] * (max_train_card - ni) , xi))
        X_train[i] = xi
        y_train[i] = yi
    return X_train, y_train


def create_test_dataset(ni_test=10_000, max_test_card=100):
    ############## Task 2
    X_test = []
    y_test = []

    for card in range(5, max_test_card + 1, 5):
        Xi_test = np.zeros((ni_test, card))
        yi_test = np.zeros(ni_test)
        for i in range(ni_test):
            ni = np.random.randint(1, card+1)
            xi = np.random.choice(np.arange(1, 11), size=ni, replace=True)
            yi = np.unique(xi).sum()
            xi = np.concatenate(([0] * (card - ni) , xi))
            Xi_test[i] = xi
            yi_test[i] = yi
        X_test.append(Xi_test)
        y_test.append(yi_test)

    return X_test, y_test


if __name__ == "__main__":
    X_train, y_train = create_train_dataset(10, 10)
    X_test, y_test = create_test_dataset(2, 20)

    print(X_train)
    print(y_train)

    print(X_test)
    print(y_test)