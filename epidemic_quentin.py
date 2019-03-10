import networkx as nx
import numpy as np
import scipy
import random
from random import randint
import matplotlib.pyplot as plt


GRAPH_FILE_PATH = 'Wiki-Vote.txt'

def nb_infected(G):
    nb=0
    for j in G.nodes():
        if int(G.node[j]['infected']) == 1:
            nb += 1
    return nb

def nb_vaccinated(G):
    nb=0
    for j in G.nodes():
        if int(G.node[j]['infected']) == 2:
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

def create_infection_vector(G,x, mode, P):

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
        print("Pagerank algo enabled")
        #pagerank = nx.pagerank(G, alpha=0.85, max_iter=100000000)
        pagerank = page_rank(P)
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

#adjacency_matrix = nx.to_numpy_matrix(G)
adjacency_matrix = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).todense()

transition_matrix = np.apply_along_axis(adjacency_to_transition, axis=1, arr=adjacency_matrix)
transition_matrix = transition_matrix.transpose()


# Initialisation des paramètres de la simulation

time = 80     # nombre d'itérations de la simulation
x = 0.05       # pourcentage d'individus aléatoirement infectés initialement = pourcentage d'individus vaccinés initialement
v = 0.2        # probabilité à chaque itération d'un individu de transmettre l'infection à un voisin
gamma = 0.24   # probabilité à chaque itération d'un individu de guérir de l'infection
alpha = 0.8
delta = 1 - alpha   # probabilité d'infecté un non-voisin

G = create_infection_vector(G, x, "pagerank", transition_matrix)

nb_infect = []
times = []

len_G = len(G.nodes())
print("INFECTION VECTOR : ", list(G.nodes(data=True)))

random.seed(5342876)


for i in range(0,time):

    nb = nb_infected(G)
    nb_v = nb_vaccinated(G)
    times.append(i)
    nb_infect.append(nb)

    print('-------- ITERATION ',i, '--------')
    print("NB INFECTED : ", nb)

    for j in G.nodes():

        # Contamination des voisins
        voisins = list(G.neighbors(j))
        if len(voisins) > 0:
            for voisin in voisins:
                rand = np.random.uniform(0, 1)
                if rand <= v and int(G.node[voisin]['infected']) != 2:
                    G.node[voisin]['infected'] = 1


        # ===============
        # NOEUD INFECTE
        if int(G.node[j]['infected']) == 1:

            # CONTAMINER SES NON-VOISINS
            neighb = list(G.neighbors(j))
            rand = np.random.uniform(0, 1) # proba d'infecter des non-voisins
            non_neighb = list(set(G.nodes()) - set(neighb) - {j}) # non-voisins du noeud
            if rand <= delta:
                rand = randint(0, len(non_neighb) - 1)
                if int(G.node[non_neighb[rand]]['infected']) != 2:
                    G.node[non_neighb[rand]]['infected'] = 1

    for j in G.nodes():
        # SE GUERIR SOIS MEME
        rand = np.random.uniform(0, 1)
        if rand <= gamma:
            G.node[j]['infected'] = 0


plt.plot(times, nb_infect)
plt.show()

