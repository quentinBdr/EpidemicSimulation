import networkx as nx
import numpy as np
import scipy
import random
import matplotlib.pyplot as plt


GRAPH_FILE_PATH = 'facebook/107.edges'

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
    # calcul du nombre d'infectés initial
    percent = int(len(G) * x)
    if (percent < 1):
        percent = 1

    # creation du vecteur
    vect = []
    for i in range(0,(len(G)-percent)):
        vect.append(0)

    # ajout des infectés aléatoirement dans le vecteur
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

# Initialisation des paramètres de la simulation

time = 100     # nombre d'itérations de la simulation
x = 0.05       # pourcentage d'individus aléatoirement infectés initialement = pourcentage d'individus vaccinés initialement
v = 0.2        # probabilité à chaque itération d'un individu de transmettre l'infection à un  voisin
gamma = 0.24   # probabilité à chaque itération d'un individu de guérir de l'infection

infect_vect = create_infection_vector(G,x)
print("INFECTION VECTOR (SIZE:",len(infect_vect),"): ",infect_vect)

nb_infect = []
t = []

print("LIST NODES (",len(G),"): ",list(G.nodes))
for i in range(0,time):
    for j in range(0, len(infect_vect)):
        #print("it",j)
        if infect_vect[j] == 1:
            rand = random.uniform(0, 1)
            if(rand <= 0.24): # probabilité de guérir
                infect_vect[j] = 0

            neighb = list(G.neighbors(j+1))
            print("NEIGHBORS OF ", j+1, ":", neighb)
            if(len(neighb) > 0):
                for k in neighb:
                    rand = random.uniform(0, 1)
                    print("    TEST FOR NODE ",k)
                    if(rand <= 0.2 and infect_vect[k] == 0): # probabilité d'infecter un voisin sain
                        #print("    NODE ",k," HAS BEEN INFECTED")
                        infect_vect[k] == 1

    for j in range(0, len(infect_vect)):
        nb = 0
        if(infect_vect[j] == 1):
            nb += 1

    t.append(time)
    nb_infect.append(nb)

plt.plot(nb_infect, t)
plt.show()

