import matplotlib.pyplot as plt
import numpy as np


weights_mode = {0: 'normal', 1: '1or0', 2: 'd-1', 3: 'd-2'}
for w_o in [0, 1, 2, 3]:
    for n_comps in [32, 64, 128]:
        filename = f"G:\\Mestrado\\BD\\data\\dbscan_epsilon\\distances_{n_comps}dim_{weights_mode[w_o]}weight.npy"
        with open(filename, "rb") as f:
            distances = np.load(f)
        print(w_o, n_comps)
        plt.plot(distances)
        plt.show()
        plt.plot(distances)
        plt.savefig(f"G:\\Mestrado\\BD\\data\\dbscan_epsilon\\distances_{n_comps}dim_{weights_mode[w_o]}weight.png")
        plt.clf()
