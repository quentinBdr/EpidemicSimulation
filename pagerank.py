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


# Generation de la matrice d'adjacence
#adjacency_matrix = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).todense()
#print("ADJACENCY MATRIX : \n",adjacency_matrix)

# Generation de la matrice de transition
#transition_matrix = np.apply_along_axis(adjacency_to_transition, axis=1, arr=adjacency_matrix)
# on transpose pour qu'elle soit indéxé en colonne et non pas en ligne
#transition_matrix = transition_matrix.transpose()
#print("\n\nTRANSITION MATRIX : \n",transition_matrix)

# Algo Page Rank
#pagerank = page_rank(transition_matrix)
#print("\n\n Page Rank : \n", pagerank)
non_neighb = list(set(G.nodes) - set(neighb) - {j}) # non-voisins du noeud
                if rand <= delta:
                    rand = randint(0, len(non_neighb) - 1)
                    if G.node[non_neighb[rand]]['infected'] < 2:
                        G.node[non_neighb[rand]]['infected'] = 1
                    #print("NON VOISIN ", non_neighb[rand], " INFECTE")
