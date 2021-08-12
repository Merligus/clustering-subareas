from os import listdir
from os.path import isfile, join

import numpy as np

import bz2
import gzip
import lzma
import pickle
import brotli

path = "data/"
data = {}
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
for filename in onlyfiles:
    p_index = filename.rfind('.')
    ext = filename[p_index+1:]
    if ext == "npy":
        filename_no_ext = filename[:p_index]
        if filename_no_ext != "graph_nao_direcionadounion":
            with open(f'{path}/{filename}', "rb") as f:
                data[filename_no_ext] = np.load(f)

    # elif ext == "pickle":
    #     with open(f'{path}/{filename}', 'rb') as handle:
    #         data = pickle.load(handle)

    #     filename_no_ext = filename[:p_index]
    #     with lzma.open(f'{path}/lzma/{filename_no_ext}.xz', "wb") as f:
    #         pickle.dump(data, f)
        
        # with lzma.open(f'{path}/lzma/{filename_no_ext}.xz', "rb") as f:
        #     data_l = pickle.load(f)
    #     if data != data_l:
    #         print(filename)

filename_no_ext = "graph_nao_direcionadounion"
np.savez_compressed(f'{path}/savez_compressed/{filename_no_ext}', **data)
data_l = np.load(f'{path}/savez_compressed/{filename_no_ext}.npz')
for name in data:
    if not np.array_equal(data[name], data_l[name]):
        print(filename)