import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn import metrics

class Agglomerative:

    def __init__(self, mode, n_clusters=1, check_symmetry=False):
        self.n_clusters = n_clusters
        self.mode = mode
        self.check_symmetry = check_symmetry

    def fit(self, adj_mat, authors_sets, debug=False, metrics_it=1, ground_truth={}, max_iter=np.inf):
        if self.check_symmetry:
            if not np.allclose(adj_mat, adj_mat.T):
                raise TypeError('Adjacency matrix must be symmetric')

        n_samples = adj_mat.shape[0]
        if self.n_clusters > n_samples or self.n_clusters < 1:
            raise TypeError(f'n_clusters must be in [1, {n_samples}]. n_clusters = {self.n_clusters}')

        if metrics_it < 1:
            raise TypeError(f'metrics_it must be greater or equal 1. metrics_it = {metrics_it}')

        self.labels_ = np.array([i for i in range(adj_mat.shape[0])])
        # dictionary to get the real index of the node
        # leafs: [0, n_samples-1]
        # merged: > n_samples
        nodes = {}
        for v in range(n_samples):
            nodes[v] = v
        # sets of clusters to determine the labels
        labels_sets = []
        for v in range(n_samples):
            labels_sets.append({v})
        i = 0
        distances = []
        children = []
        metrics_l = []
        true_index = {}
        for index in ground_truth:
            true_index[index] = index
        while i < n_samples - self.n_clusters and i < max_iter:
            # find the vertices of max similarity
            v1, v2 = np.unravel_index(np.argmax(adj_mat, axis=None), adj_mat.shape)
            if debug:
                print(f'adj_mat[{v1}, {v2}] = {adj_mat[v1, v2]}')
            # v1 and v2 must be gathered in a ground truth vertice
            if v2 in ground_truth:
                v1, v2 = v2, v1
                if v2 in ground_truth:
                    for vid in labels_sets[v2]:
                        if vid in ground_truth:
                            true_index[vid] = v1
            if debug:
                print(f'v1={v1} v2={v2}')
            # update the set of authors of the new vertice (v1 + v2)
            authors_sets[v1] = authors_sets[v1].union(authors_sets[v2])
            # update their labels
            labels_sets[v1] = labels_sets[v1].union(labels_sets[v2])
            if debug:
                print(f'new set: {labels_sets[v1]}')
            # update children to include the new merge
            children.append([nodes[v1], nodes[v2]])
            if debug:
                print(f'children: {children[-1][0]} com {children[-1][1]}')
            nodes[v1] = n_samples + i
            i += 1
            distances.append(adj_mat[v1, v2])
            if self.mode == 'union':
                # update the similarity and the set of authors between the new vertice and all others
                for v in range(n_samples):
                    if adj_mat[v1, v] >= 0:
                        set_v = authors_sets[v]
                        set_v1v2 = authors_sets[v1]
                        # common authors between v and v1+v2
                        common = len(set_v1v2.intersection(set_v))
                        # update the similarity
                        adj_mat[v1, v] = adj_mat[v, v1] = common/(len(set_v) + len(set_v1v2) - common)
            elif self.mode == 'max':
                # update the similarity and the set of authors between the new vertice and all others
                for v in range(n_samples):
                    if adj_mat[v1, v] >= 0:
                        set_v = authors_sets[v]
                        set_v1v2 = authors_sets[v1]
                        # common authors between v and v1+v2
                        common = len(set_v1v2.intersection(set_v))
                        # update the similarity
                        adj_mat[v1, v] = adj_mat[v, v1] = common/max(len(set_v), len(set_v1v2))
            elif self.mode == 'mean':
                # update the similarity and the set of authors between the new vertice and all others
                for v in range(n_samples):
                    if adj_mat[v1, v] >= 0:
                        set_v = authors_sets[v]
                        set_v1v2 = authors_sets[v1]
                        # common authors between v and v1+v2
                        common = len(set_v1v2.intersection(set_v))
                        # update the similarity
                        denom = (len(set_v) + len(set_v1v2)) / 2
                        adj_mat[v1, v] = adj_mat[v, v1] = common/denom
            elif self.mode == 'none':
                # update the similarity and the set of authors between the new vertice and all others
                for v in range(n_samples):
                    if adj_mat[v1, v] >= 0:
                        set_v = authors_sets[v]
                        set_v1v2 = authors_sets[v1]
                        # common authors between v and v1+v2
                        common = len(set_v1v2.intersection(set_v))
                        # update the similarity
                        adj_mat[v1, v] = adj_mat[v, v1] = common

            # removing
            adj_mat[:, v2] = -np.inf
            adj_mat[v2, :] = -np.inf
            authors_sets[v2] = {-1}
            nodes[v2] = -1
            labels_sets[v2] = {-1}
            # if metrics > 0, check metrics every <metrics> iterations
            if len(ground_truth) > 0 and i % metrics_it == 0:
                labels_true = []
                labels_pred = []
                for vid in true_index:
                    if vid in labels_sets[true_index[vid]]:
                        labels_true.append(ground_truth[vid])
                        labels_pred.append(true_index[vid])
                    else:
                        raise TypeError(f'vid:{vid} should be in labels_sets[{true_index[vid]}] but it is not.')
                ARS = metrics.adjusted_rand_score(labels_true, labels_pred)
                AMIS = metrics.normalized_mutual_info_score(labels_true, labels_pred)
                HS = metrics.homogeneity_score(labels_true, labels_pred)
                CS = metrics.completeness_score(labels_true, labels_pred)
                VMS = metrics.v_measure_score(labels_true, labels_pred)
                FMS = metrics.fowlkes_mallows_score(labels_true, labels_pred)
                if debug:
                    print(50*'*')
                    print(f'Iteration {i}')
                    print(f'Adjusted Rand index: {ARS:.2f}')
                    print(f'Normalized Mutual Information: {AMIS:.2f}')
                    print(f'Homogeneity: {HS:.2%}')
                    print(f'Completeness: {CS:.2%}')
                    print(f'V-measure: {VMS:.2%}')
                    print(f'Fowlkes-Mallows: {FMS:.2%}')
                    print(50*'*')
                metrics_l.append([ARS, AMIS, HS, CS, VMS, FMS])
            np.fill_diagonal(adj_mat, 0)
        
        c = 0
        for set_c in labels_sets:
            if -1 not in set_c:
                for v in set_c:
                    self.labels_[v] = c
                c += 1
        self.children_ = np.array(children)
        self.distances_ = np.array(distances)
        self.distances_ = np.max(self.distances_) - self.distances_
        self.metrics_ = np.array(metrics_l)
        return self

    # Code from https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html#sphx-glr-auto-examples-cluster-plot-agglomerative-dendrogram-py
    def plot_dendrogram(self, **kwargs):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(self.children_.shape[0])
        n_samples = len(self.labels_)
        for i, merge in enumerate(self.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([self.children_, self.distances_,
                                        counts]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)
