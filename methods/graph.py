import numpy as np


def Graph(network, p):

    network = network - np.diag(np.diag(network))
    rows, cols = network.shape
    PNN = np.zeros((rows, cols))
    graph = np.zeros((rows, cols))
    sort_network = np.sort(network, axis=1)[:, ::-1]
    idx = np.argsort(network, axis=1)[:, ::-1]

    for i in range(rows):
        PNN[i, idx[i, :p]] = sort_network[i, :p]

    for i in range(rows):
        idx_i = np.where(PNN[i, :])[0]

        for j in range(rows):
            idx_j = np.where(PNN[j, :])[0]

            if j in idx_i and i in idx_j:
                graph[i, j] = 1
            elif j not in idx_i and i not in idx_j:
                graph[i, j] = 0
            else:
                graph[i, j] = 0.5

    return graph






