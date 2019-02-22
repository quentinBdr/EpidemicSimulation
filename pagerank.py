import numpy as np
import scipy
from random import randint

def page_rank_numpy(G, P):
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    ind = eigenvalues.argsort()
    largest = np.array(eigenvectors[:, ind[-1]]).flatten().real
    norm = float(largest.sum())
    return dict(zip(G, map(float, largest / norm)))

def page_rank_power(A, iter):
    n = A.shape[1]
    A = A.reshape(n, n)
    x0 = np.ones(n)
    for i in range(iter):
        x0 = np.dot(A, x0)
        x0 = x0.reshape(n,1)
        x0 = x0 / np.linalg.norm(x0, 1)
    return x0

