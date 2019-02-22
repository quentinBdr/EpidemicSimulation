import networkx as nx
import numpy as np
import scipy
import random


GRAPH_FILE_PATH = 'facebook/0.edges'

# Utilise la matrice d'adjacence pour trouver la transition
def adjacency_to_transition( row ):
    degree=0
    for x in np.nditer(row):
        if x > 0:
            degree+=1
    return np.divide(row, degree)


# Utilise numpy pour trouver le vecteur propre à partir de la matrice de transition
def page_rank(P):
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    ind = eigenvalues.argsort()
    largest = np.array(eigenvectors[:, ind[-1]]).flatten().real
    norm = float(largest.sum())
    return dict(zip(G, map(float, largest / norm)))

def create_infection_vector(G,x):

    # calcul du nombre de noeuds à infecter
    percent = int(len(G) * x)
    if (percent < 1):
        percent = 1

    # creation du vecteur avec que des 0
    vect = []
    for i in range(0,(len(G)-percent)):
        vect.append(0)

    for i in range(0,percent):
        vect.insert(random.randint(0,len(G)),1)

    return vect


# ======================================================

# Chargement du graphe
G = nx.read_edgelist(GRAPH_FILE_PATH, create_using=nx.DiGraph(),nodetype=int)
print("NUMBER OF NODES : ",len(G))

# Generation de la matrice d'adjacence
adjacency_matrix = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).todense()
print("ADJACENCY MATRIX : \n",adjacency_matrix)

# Generation de la matrice de transition
transition_matrix = np.apply_along_axis(adjacency_to_transition, axis=1, arr=adjacency_matrix)
# on transpose pour qu'elle soit indéxé en colonne et non pas en ligne
transition_matrix = transition_matrix.transpose()
print("\n\nTRANSITION MATRIX : \n",transition_matrix)

# Algo Page Rank
pagerank = page_rank(transition_matrix)
print("\n\n Page Rank : \n", pagerank)

# Simulation Ici hihi

# Initialement
x = 0.05       # pourcentage d'individus aléatoirement infectés = pourcentage d'individus vaccinés

# Pour chaque itération
v = 0.2        # probabilité de transmettre l'infection à chaque individu
gamma = 0.24   # probabilité de guérir de l'infection

print("\nNUMBER OF INFECTED INITIALLY :",int(x*len(G)))
infect_vect = create_infection_vector(G,x)
print("INFECTION VECTOR : ",infect_vect)

