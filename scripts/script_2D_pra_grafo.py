import random
import csv
import matplotlib.pyplot as plt
import numpy as np

# from sklearn.manifold import MDS
from ..algoritmos.smacof import MDS

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

def show_communities_length(labels):
    d = {}
    for label in labels:
        if label in d:
            d[label] += 1
        else:
            d[label] = 1
    print(d)

    # creating VC to fit in info() function
    clusters = max(labels) + 1
    VC = []
    for c in range(clusters):
        VC.append((f'{c+1}.', []))
    for v, label in enumerate(labels):
        VC[label][1].append(v)
    return VC

mode = 'union'
RANDOM_STATE = 5
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
diretorio = "G:\Mestrado\BD\data\\testes\\MDS\\"

x = []
y = []
with open(f'{diretorio}banco_sintetico.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        x.append(float(row[0]))
        y.append(float(row[1]))
    
plt.scatter(x, y)
plt.savefig(f'{diretorio}plots\\banco_sintetico_plot.png')

distance = np.zeros((len(x), len(x)))
for i, t in enumerate(zip(x, y)):
    xi, yi = t[0], t[1]
    for j, s in enumerate(zip(x, y)):
        xj, yj = s[0], s[1]
        distance[i, j] = distance[j, i] = np.sqrt((xj - xi)**2 + (yj - yi)**2)

weights_mode = {0: 'normal', 1: '1or0', 2: 'd-1', 3: 'd-2'}
for w_o in [0, 1, 2, 3]:
    for n_comps in [2, 3, 4, 5, 6, 7]:
        embedders = [MDS(ndim=n_comps, weight_option=weights_mode[w_o])] # TSNE(n_components=n_comps, metric='precomputed', random_state=RANDOM_STATE)]
        for embedding in embedders:
            # Embedding
            mds_model = embedding.fit(distance) # shape = journals x n_components
            X_transformed = mds_model['conf']

            for n_clus in [2, 4]:
                # Clustering
                # K-means
                k_means = KMeans(n_clusters=n_clus, algorithm='elkan', random_state=RANDOM_STATE)
                k_means.fit(X_transformed)
                print(f'MDS KMeans weights={weights_mode[w_o]} n_clusters={n_clus} n_components = {n_comps}')
                plt.clf()
                plt.scatter(x, y, c=k_means.labels_)
                plt.savefig(f'{diretorio}plots\\banco_sintetico_plot_kmeans_c={n_clus}_w={weights_mode[w_o]}_d={n_comps}.png')
                VC = show_communities_length(k_means.labels_)
                print(k_means.labels_)
                # def ind_value(e):
                #     s = 0
                #     list_of_ind = e[0][:-1].split('.')
                #     while len(list_of_ind) < 5:
                #         list_of_ind.append('0')
                #     list_of_ind.reverse()
                #     for i, xi in enumerate(list_of_ind):
                #         s += float(xi)*1e4**(i)
                #     return s
                # VC.sort(key=ind_value)
                print(VC)

                # GMM
                gmm_c = GaussianMixture(n_components=n_clus, random_state=RANDOM_STATE)
                gmm_c.fit(X_transformed)
                print(f'MDS GMM weights={weights_mode[w_o]} n_clusters={n_clus} n_components = {n_comps} {"converged" if gmm_c.converged_ else "did not converge"}')
                gmm_labels = gmm_c.predict(X_transformed)
                plt.clf()
                plt.scatter(x, y, c=gmm_labels)
                plt.savefig(f'{diretorio}plots\\banco_sintetico_plot_gmm_c={n_clus}_w={weights_mode[w_o]}_d={n_comps}.png')
                VC = show_communities_length(gmm_labels)
                print(gmm_labels)
                print(VC)

                # VBGMM
                vbgmm_c = BayesianGaussianMixture(n_components=n_clus, weight_concentration_prior=0.01, max_iter=1700, random_state=RANDOM_STATE)
                vbgmm_c.fit(X_transformed)
                print(f'MDS VBGMM weights={weights_mode[w_o]} n_clusters={n_clus} n_components = {n_comps} {"converged" if vbgmm_c.converged_ else "did not converge"}')
                vbgmm_labels = vbgmm_c.predict(X_transformed)
                plt.clf()
                plt.scatter(x, y, c=vbgmm_labels)
                plt.savefig(f'{diretorio}plots\\banco_sintetico_plot_vbgmm_c={n_clus}_w={weights_mode[w_o]}_d={n_comps}.png')
                VC = show_communities_length(vbgmm_labels)
                print(vbgmm_labels)
                print(VC)

            del X_transformed