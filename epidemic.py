import networkx as nx
import numpy as np
import scipy
import random
from random import randint
import matplotlib.pyplot as plt


GRAPH_FILE_PATH = 'p2p.txt'

# Utilise la matrice d'adjacence pour trouver la transition
def adjacency_to_transition( row ):
    degree=0
    for x in np.nditer(row):
        if x > 0:
            degree+=1

    if degree == 0:
        return row
    else:
        return np.divide(row, degree)


# Utilise numpy pour trouver le vecteur propre à partir de la matrice de transition
def page_rank(P):
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    ind = eigenvalues.argsort()
    largest = np.array(eigenvectors[:, ind[-1]]).flatten().real
    norm = float(largest.sum())
    return dict(zip(G, map(float, largest / norm)))

def create_infection_vector(G,x):

    # calcul du nombre d'infectés initial
    len_G = len(G)
    percent = int(len_G * x)
    if percent < 1:
        percent = 1

    # creation du vecteur
    nx.set_node_attributes(G, 0, 'infected')

    # ajout des vaccinés selon le résultat de pagerank dans le vecteur
    pagerank = nx.pagerank(G, alpha=0.85)
    for i in range(0, percent):
        maxPage = max(pagerank, key=pagerank.get)
        G.node[maxPage]['infected'] = 2
        pagerank.pop(maxPage)

    # ajout des infectés aléatoirement dans le vecteur
    for i in range(0,percent):
        rand = randint(0, len_G - 1)
        if G.has_node(rand):
            if int(G.node[rand]['infected']) == 0:
                G.node[rand]['infected'] = 1
            else:
                i -= 1
        else:
            i -= 1



    return G

# ======================================================

# Chargement du graphe
G = nx.read_edgelist(GRAPH_FILE_PATH, create_using=nx.DiGraph(),nodetype=int)
print(nx.info(G))

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

# Simulation Ici hihi

# Initialisation des paramètres de la simulation

time = 300     # nombre d'itérations de la simulation
x = 0.05       # pourcentage d'individus aléatoirement infectés initialement = pourcentage d'individus vaccinés initialement
v = 0.2        # probabilité à chaque itération d'un individu de transmettre l'infection à un voisin
gamma = 0.15   # probabilité à chaque itération d'un individu de guérir de l'infection
alpha = 0.8
delta = 1 - alpha

G = create_infection_vector(G,x)

nb_infect = []
t = []

len_G = len(G)
print("INFECTION VECTOR : ", list(G.nodes(data=True)))

for i in range(0,time):

    nb = 0
    for j in range(0, len_G):
        if G.has_node(j):
            if int(G.node[j]['infected']) == 1:
                nb += 1

    t.append(i)
    nb_infect.append(nb)

    print('-------- ITERATION ',i, '--------')
    print("NB INFECTED : ", nb)

    for j in range(0, len_G):
        if G.has_node(j):
            if int(G.node[j]['infected']) == 1:
                rand = random.uniform(0, 1)
                if rand <= gamma: # probabilité de guérir
                    G.node[j]['infected'] = 0

                neighb = list(G.neighbors(j))
                #print("NEIGHBORS OF ", j, ":", neighb)
                if len(neighb) > 0:
                    for k in list(neighb):
                        rand = random.uniform(0, 1)
                        if rand <= v and int(G.node[k]['infected']) < 1: # probabilité d'infecter un voisin sain non vacciné
                            #print("    NODE ",k," HAS BEEN INFECTED")
                            G.node[k]['infected'] == 1

                rand = random.uniform(0, 1) # proba d'infecter des non-voisins
                non_neighb = list(set(G.nodes) - set(neighb) - {j}) # non-voisins du noeud
                if rand <= delta:
                    rand = randint(0, len(non_neighb) - 1)
                    if G.node[non_neighb[rand]]['infected'] < 2:
                        G.node[non_neighb[rand]]['infected'] = 1
                    #print("NON VOISIN ", non_neighb[rand], " INFECTE")

plt.plot(t, nb_infect)
plt.show()

