import networkx as nx
import numpy as np
import scipy
import random
from random import randint
import matplotlib.pyplot as plt


GRAPH_FILE_PATH = 'Wiki-Vote.txt'

def nb_infected(G):
    nb=0
    for j in range(0, len(G)):
        if G.has_node(j):
            if int(G.node[j]['infected']) == 1:
                nb += 1
    return nb

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

def create_infection_vector(G,x, mode):

    # calcul du nombre d'infectés initial
    len_G = len(G)
    nb_to_be_infected = int(len_G * x)
    if nb_to_be_infected < 1:
        nb_to_be_infected = 1

    nb_to_be_vaccinated = nb_to_be_infected

    # creation du vecteur
    nx.set_node_attributes(G, 0, 'infected')

    # ajout des vaccinés selon le résultat de pagerank dans le vecteur
    if mode == "pagerank":
        pagerank = nx.pagerank(G, alpha=0.85)
        for i in range(0, nb_to_be_vaccinated):
            maxPage = max(pagerank, key=pagerank.get)
            G.node[maxPage]['infected'] = 2
            pagerank.pop(maxPage)

    i=0
    while i < nb_to_be_infected:
        rand = randint(0, len_G - 1)
        if G.has_node(rand):
            if int(G.node[rand]['infected']) == 0:
                G.node[rand]['infected'] = 1
            else:
                i -= 1
        else:
            i -= 1
        i += 1
    return G

# ==================================================================================================
# ==================================================================================================
# ==================================================================================================

# Chargement du graphe
G = nx.read_edgelist(GRAPH_FILE_PATH, create_using=nx.DiGraph(),nodetype=int)
print(nx.info(G))

# Initialisation des paramètres de la simulation

time = 140     # nombre d'itérations de la simulation
x = 0.05       # pourcentage d'individus aléatoirement infectés initialement = pourcentage d'individus vaccinés initialement
v = 0.2        # probabilité à chaque itération d'un individu de transmettre l'infection à un voisin
gamma = 0.24   # probabilité à chaque itération d'un individu de guérir de l'infection
alpha = 0.8
delta = 1 - alpha

G = create_infection_vector(G, x, "normal")

nb_infect = []
times = []

len_G = len(G)
print("INFECTION VECTOR : ", list(G.nodes(data=True)))

for i in range(0,time):

    nb = nb_infected(G)
    times.append(i)
    nb_infect.append(nb)

    print('-------- ITERATION ',i, '--------')
    print("NB INFECTED : ", nb)

    for j in range(0, len_G):
        if G.has_node(j):
            # Contamination des voisins
            neighb = list(G.neighbors(j))
            if len(neighb) > 0:
                for k in list(neighb):
                    rand = random.uniform(0, 1)
                    if rand <= v and int(G.node[k]['infected']) == 0:
                        G.node[k]['infected'] == 1

            # ===============
            # NOEUD INFECTE
            if int(G.node[j]['infected']) == 1:

                # Proba de guérir
                rand = random.uniform(0, 1)
                if rand <= gamma:
                    G.node[j]['infected'] = 0

                rand = random.uniform(0, 1) # proba d'infecter des non-voisins
                non_neighb = list(set(G.nodes) - set(neighb) - {j}) # non-voisins du noeud
                if rand <= delta:
                    rand = randint(0, len(non_neighb) - 1)
                    if G.node[non_neighb[rand]]['infected'] < 2:
                        G.node[non_neighb[rand]]['infected'] = 1

plt.plot(times, nb_infect)
plt.show()

