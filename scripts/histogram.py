import numpy as np
import matplotlib.pyplot as plt

with open('G:\\Mestrado\\BD\\data\\adj_mat.npy', 'rb') as f:
    adj_mat = np.load(f)

# adj_mat = adj_mat + 1e-10
# distance = 1/adj_mat
distance = adj_mat.max() - adj_mat
np.fill_diagonal(distance, 0)

dist_flatten = np.array(distance).flatten()
# max - adj_mat: (0.36, 0.3612)
_ = plt.hist(dist_flatten, range=(0.36, 0.3612), bins=30)
print(f"Media = {dist_flatten.mean():.2f}, Mediana = {np.median(dist_flatten):.2f}")
plt.title(f"Media = {dist_flatten.mean():.2f}, Mediana = {np.median(dist_flatten):.2f}")
plt.show()