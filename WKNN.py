import numpy as np

def WKNN_method(Y_mat, SM_mat, SD_mat, K, r):

    rows, cols = Y_mat.shape
    y_m = np.mat(np.zeros((rows, cols)))
    y_d = np.mat(np.zeros((rows, cols)))

    knn_network_m = KNN(SM_mat, K)
    for i in range(rows):
        w = np.zeros(K)
        sort_m, idx_m = np.sort(knn_network_m[i,:], axis=0)[::-1], np.argsort(knn_network_m[i,:], axis=0)[::-1]
        sum_m = np.sum(sort_m[:K])
        for j in range(K):
            w[j] = r ** j * sort_m[j]
            y_m[i,:] += w[j] * Y_mat[idx_m[j],:]
        y_m[i,:] /= sum_m

    knn_network_d = KNN(SD_mat, K)
    for i in range(cols):
        w = np.zeros(K)
        sort_d, idx_d = np.sort(knn_network_d[i,:], axis=0)[::-1], np.argsort(knn_network_d[i,:], axis=0)[::-1]
        sum_d = np.sum(sort_d[:K])

        for j in range(K):
            w[j] = r ** j * sort_d[j]
            y_d[:, i] += w[j] * Y_mat[:, idx_d[j]]
        y_d[:, i] /= sum_d

    a1 = 1
    a2 = 1
    y_md = (y_m * a1 + y_d * a2) / (a1 + a2)

    Y_mat_new = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            Y_mat_new[i, j] = max(Y_mat[i, j], y_md[i, j])

    return Y_mat_new

def KNN(network, k):
    rows, cols = network.shape
    network = network - np.diag(np.diag(network))

    knn_network = np.zeros((rows, cols))
    sort_network, idx = np.sort(network, axis=1)[:,::-1], np.argsort(network, axis=1)[:,::-1]
    for i in range(rows):
        knn_network[i, idx[i, :k]] = sort_network[i, :k]

    return knn_network

