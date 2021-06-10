import numpy as np
from scipy.cluster.hierarchy import dendrogram

class Agglomerative:

    def __init__(self, mode, n_clusters=1, check_symmetry=False):
        self.n_clusters = n_clusters
        self.mode = mode
        self.check_symmetry = check_symmetry

    def fit(self, adj_mat, authors_sets, debug=False):
        if self.check_symmetry:
            if not np.allclose(adj_mat, adj_mat.T):
                raise TypeError('Adjacency matrix must be symmetric')

        n_samples = adj_mat.shape[0]
        if self.n_clusters > n_samples or self.n_clusters < 1:
            raise TypeError(f'n_clusters must be in [1, {V}]. n_clusters = {self.n_clusters}')

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
        max_adj_mat = adj_mat.max()
        while i < n_samples - self.n_clusters:
            # find the vertices of max similarity
            v1, v2 = np.unravel_index(np.argmax(adj_mat, axis=None), adj_mat.shape)
            if debug:
                print(f'adj_mat[{v1}, {v2}] = {adj_mat[v1, v2]}')
            # v1 must be less or equal than v2
            v1, v2 = min(v1, v2), max(v1, v2)
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
            distances.append(max_adj_mat - adj_mat[v1, v2])
            if self.mode == 'union':
                # update the similarity and the set of authors between the new vertice and all others
                for v in range(n_samples):
                    set_v = authors_sets[v]
                    set_v1v2 = authors_sets[v1]
                    # common authors between v and v1+v2
                    common = len(set_v1v2.intersection(set_v))
                    # update the similarity
                    adj_mat[v1, v] = adj_mat[v, v1] = common/(len(set_v) + len(set_v1v2) - common)
            elif self.mode == 'min':
                # update the similarity and the set of authors between the new vertice and all others
                for v in range(n_samples):
                    set_v = authors_sets[v]
                    set_v1v2 = authors_sets[v1]
                    # common authors between v and v1+v2
                    common = len(set_v1v2.intersection(set_v))
                    # update the similarity
                    adj_mat[v1, v] = adj_mat[v, v1] = common/min(len(set_v), len(set_v1v2))
            elif self.mode == 'mean':
                # update the similarity and the set of authors between the new vertice and all others
                for v in range(n_samples):
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

            np.fill_diagonal(adj_mat, 0)
        
        for c, set_c in enumerate(labels_sets):
            for v in set_c:
                self.labels_[v] = c
        self.children_ = np.array(children)
        self.distances_ = np.array(distances)
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
