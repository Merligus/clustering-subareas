import numpy as np

from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

filename = "../data/graph_nao_direcionadounion.npy"
with open(filename, "rb") as f:
    adj_mat = np.load(f)
distance = adj_mat.max() - adj_mat
np.fill_diagonal(distance, 0)
del adj_mat

d = {0: 'normal', 1: '1or0', 2: 'd-1', 3: 'd-2'}
for w_o in [0, 1, 2, 3]:
    for n_comps in [2, 3, 4, 5, 6, 7, 32, 64, 128]:
        print(f'Processando {d[w_o]}weight e {n_comps}dim')
        if w_o == 0 and n_comps == 2:
            continue
        print(f'calculando X_transformed')
        embedders = MDS(n_components=n_comps, dissimilarity='precomputed', metric=True, random_state=5, weight_option=w_o)
        # Embedding
        X_transformed = embedders.fit_transform(distance) # shape = journals x n_components
        print(f'salvando X_transformed')
        with open(f"../data/distance_embedded/X_transformed_{n_comps}dim_{d[w_o]}weight.npy", "wb") as f:
            np.save(f, X_transformed)

        print(f'calculando distances')
        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(X_transformed)
        distances, indices = nbrs.kneighbors(X_transformed)

        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        print(f'salvando distances')
        with open(f"../data/epsilons/distances_{n_comps}dim_{d[w_o]}weight.npy", "wb") as f:
            np.save(f, distances)

        del embedders
        del X_transformed
        del neigh
        del nbrs
        del distances
        del indices      
