import numpy as np
import matplotlib.pyplot as plt

with open('G:\\Mestrado\\BD\\data\\graph_nao_direcionadounion_2010_cut0.2.npy', 'rb') as f:
    adj_mat = np.load(f)

# adj_mat = adj_mat + 1e-10
# distance = 1/adj_mat
distance = adj_mat.max() + adj_mat.min() - adj_mat
np.fill_diagonal(adj_mat, 0)

dist_flatten = np.array(adj_mat).flatten()
# max - adj_mat: (0.36, 0.3612)
_ = plt.hist(dist_flatten, range=(0, 0.012), bins=30)
print(f"Media = {dist_flatten.mean():.5f}, Mediana = {np.median(dist_flatten):.5f}")
plt.title(f"Media = {dist_flatten.mean():.5f}, Mediana = {np.median(dist_flatten):.5f}")
plt.show()