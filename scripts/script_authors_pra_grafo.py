import csv
import random
import matplotlib.pyplot as plt
import numpy as np
from ..algoritmos.agglomerative import Agglomerative

mode = 'union'
RANDOM_STATE = 5
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
diretorio = "G:\Mestrado\BD\data\\testes\\Agglomerative\\"

authors_sets = []
with open(f'{diretorio}banco_sintetico.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        authors_sets.append(set())
        for author in row:
            authors_sets[-1].add(author.strip())

adj_mat = np.zeros((len(authors_sets), len(authors_sets)))
for i, a_seti in enumerate(authors_sets):
    for j, a_setj in enumerate(authors_sets):
        common = len(a_seti.intersection(a_setj))
        if mode == 'union':
            adj_mat[i, j] = adj_mat[j, i] = common/(len(a_seti) + len(a_setj) - common)
        elif mode == 'mean':
            adj_mat[i, j] = adj_mat[j, i] = common/(len(a_seti)/2 + len(a_setj)/2)
        elif mode == 'min':
            adj_mat[i, j] = adj_mat[j, i] = common/min(len(a_seti), len(a_setj))
        elif mode == 'none':
            adj_mat[i, j] = adj_mat[j, i] = common
np.fill_diagonal(adj_mat, 0)

model = Agglomerative(mode=mode).fit(adj_mat=adj_mat, authors_sets=authors_sets, debug=False)
p = 5
min_d, max_d = np.min(model.distances_[-(p):]), np.max(model.distances_[-(p):])
model.distances_[-(p):] = (model.distances_[-(p):] - min_d) / (max_d - min_d)
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
model.plot_dendrogram(truncate_mode='lastp', p=p, distance_sort=True)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.savefig(f'{diretorio}dendogram_{mode}_o.png')