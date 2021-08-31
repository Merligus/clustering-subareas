from os import listdir
from os.path import isfile, join

import numpy as np

import lzma
import pickle

path_not_compressed = "../data/not_compressed/"
path = "../data/"
np_data = {}
onlyfiles = [f for f in listdir(path_not_compressed) if isfile(join(path_not_compressed, f))]
for filename in onlyfiles:
    print(filename)
    p_index = filename.rfind('.')
    ext = filename[p_index+1:]
    if ext == "npy":
        filename_no_ext = filename[:p_index]
        with open(f'{path_not_compressed}/{filename}', "rb") as f:
            np_data[filename_no_ext] = np.load(f)

    elif ext == "pickle":
        with open(f'{path_not_compressed}/{filename}', 'rb') as handle:
            data = pickle.load(handle)

        filename_no_ext = filename[:p_index]
        with lzma.open(f'{path}/{filename_no_ext}.xz', "wb") as f:
            pickle.dump(data, f)
        
        with lzma.open(f'{path}/{filename_no_ext}.xz', "rb") as f:
            data_l = pickle.load(f)
        if data != data_l:
            print(filename)
        del data_l
        del data

filename_no_ext = "graph_nao_direcionadomean"
np.savez_compressed(f'{path}/{filename_no_ext}', **np_data)
data_l = np.load(f'{path}/{filename_no_ext}.npz')
for name in np_data:
    if not np.array_equal(np_data[name], data_l[name]):
        print(filename)