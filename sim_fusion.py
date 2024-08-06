
import numpy as np

def new_normalization(W):
    m = W.shape[0]
    p = np.zeros([m,m])
    for i in range(m):
        for j in range(m):
            if i == j:
                p[i][j] = 1/2
            elif np.sum(W[i,:])-W[i,i]>0:
                p[i][j] = W[i,j]/(2*(np.sum(W[i,:])-W[i,i]))
    return p

def KNN_kernel(S, k):
    n = S.shape[0]
    S_knn = np.zeros([n, n])
    for i in range(n):
        sort_index = np.argsort(S[i, :])
        for j in sort_index[n - k:n]:
            if np.sum(S[i, sort_index[n - k:n]]) > 0:
                S_knn[i][j] = S[i][j] / (np.sum(S[i, sort_index[n - k:n]]))
    return S_knn

def miRNA_updating(S1, S2, S3, P1, P2, P3):
    it = 0
    P = (P1 + P2 + P3) / 3
    dif = 1
    while dif > 0.00001:
        it = it + 1
        P111 = np.dot(np.dot(S1, (P2 + P3) / 2), S1.T)
        P111 = new_normalization(P111)
        P222 = np.dot(np.dot(S2, (P1 + P3) / 2), S2.T)
        P222 = new_normalization(P222)
        P333 = np.dot(np.dot(S3, (P1 + P2) / 2), S3.T)
        P333 = new_normalization(P333)

        P1 = P111
        P2 = P222
        P3 = P333

        P_New = (P1 + P2 + P3) / 3
        dif = np.linalg.norm(P_New - P) / np.linalg.norm(P)
        P = P_New
    print("Iter numb1", it)
    return P

def disease_updating(S1, S2, P1, P2):
    it = 0
    P = (P1 + P2) / 2
    dif = 1
    while dif > 0.00001:
        it = it + 1
        P111 = np.dot(np.dot(S1, P2), S1.T)
        P111 = new_normalization(P111)
        P222 = np.dot(np.dot(S2, P1), S2.T)
        P222 = new_normalization(P222)
        P1 = P111
        P2 = P222
        P_New = (P1 + P2) / 2
        dif = np.linalg.norm(P_New - P) / np.linalg.norm(P)
        P = P_New
    print("Iter numb2", it)
    return P
