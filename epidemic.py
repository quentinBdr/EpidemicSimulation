import networkx as nx
import numpy as np


G = nx.read_edgelist('facebook/td.txt', create_using=nx.DiGraph(),nodetype=float)


def adjacency_to_transition( row ):
    degree=0
    for x in np.nditer(row):
        if x > 0:
            degree+=1
    return np.divide(row, degree)



adjacency_matrix = nx.to_numpy_matrix(G)
print("ADJACENCY MATRIX : \n",adjacency_matrix)

transition_matrix = np.apply_along_axis(adjacency_to_transition, axis=1, arr=adjacency_matrix)
transition_matrix = transition_matrix.transpose()
print ("\n\nTRANSITION MATRIX : \n",transition_matrix)
