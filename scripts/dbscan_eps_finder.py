import numpy as np

from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

filename = "../data/distance.npy"
with open(filename, "rb") as f:
    distance = np.load(f)

d = {0: 'normal', 1: '1or0', 2: 'd-1', 3: 'd-2'}
for w_o in [0, 1, 2, 3]:
    for n_comps in [32, 64, 128]:
        embedders = MDS(n_components=n_comps, dissimilarity='precomputed', metric=True, random_state=7, weight_option=w_o)
        # Embedding
        X_transformed = embedders.fit_transform(distance) # shape = journals x n_components
        with open(f"../data/distance_embedded/X_transformed_{n_comps}dim_{d[w_o]}weight.npy", "wb") as f:
            np.save(f, X_transformed)

        neigh = NearestNeighbors(n_neighbors=2)
        nbrs = neigh.fit(X_transformed)
        distances, indices = nbrs.kneighbors(X_transformed)

        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        with open(f"../data/epsilons/distances_{n_comps}dim_{d[w_o]}weight.npy", "wb") as f:
            np.save(f, distances)
