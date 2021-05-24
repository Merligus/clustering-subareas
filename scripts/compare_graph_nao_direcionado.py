import numpy as np

name1 = 'graph_nao_direcionadounion.npy'
with open('../data/' + name1, 'rb') as f:
    adj_mat1 = np.load(f)

name2 = 'graph_nao_direcionadounion_no_proceedings.npy'
with open('../data/' + name2, 'rb') as f:
    adj_mat2 = np.load(f)

print(adj_mat1.shape)
print(adj_mat2.shape)
print(np.array_equal(adj_mat1, adj_mat2))
print(f'diferencas = {np.count_nonzero(np.where(adj_mat1 == adj_mat2, 0, 1))}')
