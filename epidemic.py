import networkx as nx
import numpy as np

def adjacency_to_transition( row ):
    degree=0
    for x in np.nditer(row):
        if x > 0:
            degree+=1
    return np.divide(row, degree)

def page_rank(P):
    # use numpy LAPACK solver
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    ind = eigenvalues.argsort()
    # eigenvector of largest eigenvalue at ind[-1], normalized
    largest = np.array(eigenvectors[:, ind[-1]]).flatten().real
    norm = float(largest.sum())
    return dict(zip(G, map(float, largest / norm)))



# ======================================================

G = nx.read_edgelist('facebook/0.edges', create_using=nx.DiGraph(),nodetype=int)
print(G.edges())

adjacency_matrix = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).todense()
print("ADJACENCY MATRIX : \n",adjacency_matrix)

transition_matrix = np.apply_along_axis(adjacency_to_transition, axis=1, arr=adjacency_matrix)
transition_matrix = transition_matrix.transpose()
print ("\n\nTRANSITION MATRIX : \n",transition_matrix)

pagerank = page_rank(transition_matrix)
print ("\n\n Page Rank : \n", pagerank)




