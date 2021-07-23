import matplotlib
import matplotlib.pyplot as plt
from igraph import *
from nltk.featstruct import _default_fs_class
import numpy as np
import random
import pickle
import sys
import os

from sklearn import metrics
# from sklearn.manifold import MDS
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS

from agglomerative import Agglomerative
from cluster_rec import ClusterRec
from smacof import MDS


# only_ground_truth=True, only_labeled=True para descobrir o valor maximo que silhouette chega com a matriz "distance"
# only_ground_truth=False, only_labeled=True para comparar com o valor maximo do silhouette quando only_ground_truth=True
# only_labeled=False. Executa o silhouette na matriz inteira
def info(f, VC, d_ind_pra_nome, representantes, iniciais, G, distance, only_ground_truth=True, only_labeled=True, metric='precomputed'):
    def ind_value(e):
        s = 0
        list_of_ind = e[0][:-1].split('.')
        while len(list_of_ind) < 5:
            list_of_ind.append('0')
        list_of_ind.reverse()
        for i, xi in enumerate(list_of_ind):
            s += float(xi)*1e4**(i)
        return s
    VC.sort(key=ind_value)
    labels_true = []
    labels_pred = []
    labels = [0]*G.vs.indegree().__len__()
    initials = []
    initial_labels_pred = []
    for comm_ind, tuple in enumerate(VC):
        ind, comm = tuple[0], tuple[1]
        f.write(f"{comm_ind}={ind}\n")
        lista_de_jornais = []
        for v in comm:
            if G.vs[v]["initial"] > -1:
                initials.append(v)
                if only_ground_truth:
                    initial_labels_pred.append(G.vs[v]["initial"])
                else:
                    initial_labels_pred.append(comm_ind)
            labels[v] = comm_ind
            try:
                f.write(f"\t{comm_ind}={ind}:{d_ind_pra_nome[v]}\n")
                lista_de_jornais.append(d_ind_pra_nome[v])
            except:
                a = 1
        # for i in representantes:
        #     if representantes[i] in lista_de_jornais:
        #         print(f'Representante {representantes[i]} ficou em uma comunidade de tamanho {len(comm)}')
        #         break
        for i in range(len(iniciais)):
            for name in iniciais[i]:
                if name in lista_de_jornais:
                    # print(f'{name} da comunidade {i} foi para a comunidade {comm_ind}')
                    labels_true.append(i)
                    labels_pred.append(comm_ind)
    # Metrics
    # Com ground truth
    print(f'Adjusted Rand index: {metrics.adjusted_rand_score(labels_true, labels_pred):.2f}')
    print(f'Adjusted Mutual Information: {metrics.adjusted_mutual_info_score(labels_true, labels_pred):.2f}')
    print(f'Homogeneity: {metrics.homogeneity_score(labels_true, labels_pred):.2%}')
    print(f'Completeness: {metrics.completeness_score(labels_true, labels_pred):.2%}')
    print(f'V-measure: {metrics.v_measure_score(labels_true, labels_pred):.2%}')
    print(f'Fowlkes-Mallows: {metrics.fowlkes_mallows_score(labels_true, labels_pred):.2%}')
    if only_labeled: # Com ground truth
        # Apenas pegando as distancias dos jornais do ground truth
        if metric == 'precomputed':
            dist_ = distance[initials,:][:,initials]
        else:
            dist_ = distance[initials,:][:,:]
        print(f'Silhouette: {metrics.silhouette_score(dist_, initial_labels_pred, metric=metric):.2f}')
    else: # Sem ground truth
        print(f'Silhouette: {metrics.silhouette_score(distance, labels, metric=metric):.2f}')
    return labels

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

print('STARTING')

if(len(sys.argv) < 8):
    print("Falta parametros")
    exit()
elif(len(sys.argv) > 8):
    print("Muitos parametros")
    exit()
else:
    log_transf = bool(int(sys.argv[1]))
    mode = sys.argv[2] # mean, min, union
    function = sys.argv[3] # multilevel, fastgreedy, agglomerative, reciprocal, . . .
    TIMES = int(sys.argv[4])
    n_components = int(sys.argv[5])
    if sys.argv[6] != '-':
        in_name = '_' + sys.argv[6]
    else:
        in_name = ''
    nan_sub = bool(int(sys.argv[7])) # True -> adj_mat[==0] = nan

RANDOM_STATE = 5
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# 0: nao direcionado
# 1: bidirecionado
# 2: bipartido
opcao_grafo = 0

do_spanning_tree = False
do_mds = n_components > 1
cut_p = True

# Gerador no arquivo teste?
test = False

test_name = '_' + mode
if test:
    test_name += "_test"
if log_transf:
    test_name += '_' + 'log'
if n_components <= 1:
    test_name += f'_rec{TIMES}'

np.set_printoptions(threshold=np.inf)

# 0: nao direcionado
# 1: bidirecionado
# 2: bipartido
if opcao_grafo == 0:
    filename = "../data/graph_nao_direcionado" + mode + in_name + '.npy'
    with open(filename, "rb") as f:
        adj_mat = np.load(f)
    print('LOADED GRAPH')
    with open('../data/journalname' + in_name + '.pickle', 'rb') as handle:
        journalname = pickle.load(handle)
    print('LOADED JOURNAL NAMES')
    V = adj_mat.shape[0]

    ################## iGraph
    G = Graph()
    G.add_vertices(V)
    edges = []
    weights = []
    v1 = 0

    # Cortando uma porcentagem dos pesos mais baixos
    if cut_p:
        adj_flatten = adj_mat.flatten()
        k = int(adj_mat.shape[0] * adj_mat.shape[1] * 0.1)
        idx = np.argpartition(adj_flatten, k, axis=None)
        adj_flatten[idx[:k]] = 0.
        adj_mat = adj_flatten.reshape(adj_mat.shape[0], adj_mat.shape[1])

    if log_transf:
        adj_mat = np.where(adj_mat == 0, np.inf, adj_mat)
        adj_mat = np.log(adj_mat)-np.log(np.min(adj_mat)-1e-15)
    while v1 < V:
        v2 = v1 + 1
        while v2 < V:
            if adj_mat[v1, v2] > 0 and adj_mat[v1, v2] < np.inf:
                edges.append((v1, v2))
                weights.append(adj_mat[v1, v2])
            v2 += 1
        v1 += 1

    # Colocando os pesos do grafo negativos para calcular
    # o 'maximum' spanning tree
    print('ADDING EDGES')
    G.add_edges(edges)
    print('ADDING JOURNAL NAMES')
    G.vs["journalname"] = journalname

    if do_spanning_tree:
        G.es["commonauthors"] = [-w for w in weights]
        spanning_trees_IDs = G.spanning_tree(weights=G.es['commonauthors'], return_tree=False)
        print(f'IDs = {spanning_trees_IDs}')
        print(50*'*')

    if log_transf:
        min_pesos = adj_mat.min(axis=1)
        adj_mat = np.where(adj_mat == np.inf, 0, adj_mat)
        max_pesos = adj_mat.max(axis=1)
        adj_mat = np.where(adj_mat == 0, np.inf, adj_mat)
    else:
        max_pesos = adj_mat.max(axis=1)
        adj_mat = np.where(adj_mat == 0, np.inf, adj_mat)
        min_pesos = adj_mat.min(axis=1)
        adj_mat = np.where(adj_mat == np.inf, 0, adj_mat)

    # Colocando os verdadeiros pesos do grafo
    G.es["commonauthors"] = weights
    print('WEIGHTS PROCESSED')
    
elif opcao_grafo == 1: 
    G = load("../data/graph_direcionado" + test_name + "2.gml")
else:
    G = load("../data/graph_bipartido" + test_name + ".gml")
    # print(summary(G))
    # V = len(G.vs["name"])
    # for vid in range(V):
    #     print(vid, G.vs[vid]["name"], G.vs[vid]["isjournal"])
    
    # edges = G.get_edgelist()
    # for eid, e in enumerate(edges):
    #     print(e[0], "--", e[1], G.es[eid]["publications"])
    exit()

initial = { 'ai' : 0,
            'jair' : 0,
            'jar' : 0,
            'aaai' : 0,
            'ijcai' : 0,

            'bmcbi' : 1,
            'bioinformatics' : 1,
            'jcb' : 1,
            'recomb' : 1,
            'tcbb' : 1,


            'ton' : 2,
            'tcom' : 2,
            'mobicom' : 2,
            'sigcomm' : 2,
            'infocom' : 2,


            'oopsla' : 3,
            'popl' : 3,
            'pldi' : 3,
            'toplas' : 3,
            'cgo' : 3,

            'isca' : 4,
            'micro' : 4,
            'dac' : 4,
            'asplos' : 4,
            'tcad' : 4,
            'tjs' : 4,

            'tog' : 5,
            'cga' : 5,
            'tvcg' : 5,
            'siggraph' : 5,
            'vis' : 5,

            'tods' : 6,
            'vldb' : 6,
            'pods' : 6,
            'sigmod' : 6,
            'jdm' : 6,

            'tpds' : 7,
            'jpdc' : 7,
            'podc' : 7,
            'icdcs' : 7,

            'tochi' : 8,
            'ijmms' : 8,
            'umuai' : 8,
            'chi' : 8,
            'cscw' : 8,

            'ijcv' : 9,
            'tip' : 9,
            'cvpr' : 9,
            'icip' : 9,

            'ml' : 10,
            'jmlr' : 10,
            'neco' : 10,
            'nips' : 10,
            'icml' : 10,

            'isr' : 11,
            'mansci' : 11,
            'jmis' : 11,
            'ejis' : 11,
            'misq' : 11,

            'mms' : 12,
            'tmm' : 12,
            'ieeemm' : 12,
            'mm' : 12,
            'icmcs' : 12,

            'mp' : 13,
            'siamjo' : 13,
            'or' : 13,
            'informs' : 13,
            'cor' : 13,
            'dam' : 13,


            'tissec' : 14,
            'jcs' : 14,
            'ieeesp' : 14,
            'sp' : 14,
            'uss' : 14,
            'ccs' : 14,

            'tse' : 15,
            'tosem' : 15,
            'icse' : 15,
            'ese' : 15,
            'tacas' : 15,

            'jacm' : 16,
            'siamcomp' : 16,
            'stoc' : 16,
            'focs' : 16,
            'soda' : 16}

fast_journalname = {}
for journal in journalname:
    fast_journalname[journal] = True

remove_list = []
for journal in initial:
    if journal not in fast_journalname:
        remove_list.append(journal)

for journal_r in remove_list:
    initial.pop(journal_r, None)

# lista de listas
lista_iniciais = []
for i in range(max(initial.values())+1):
    lista_iniciais.append([])
for name in initial:
    lista_iniciais[initial[name]].append(name)
print('INITIAL')

# Para cada comunidade i haverá um jornal/conferência representante
lideres = {}
for journal in initial:
    lideres[initial[journal]] = journal
#print(f'Representantes de cada comunidade = {lideres}')
print(50*'*')

V = len(G.vs.indegree())
index_to_journalname = {}
index_to_ground_truth = {}
chosen = []
for vid in range(V):
    index_to_journalname[vid] = G.vs[vid]["journalname"]
    if G.vs[vid]["journalname"] in initial:
        G.vs[vid]["initial"] = initial[G.vs[vid]["journalname"]]
        G.vs[vid]["fixed"] = True
        index_to_ground_truth[vid] = initial[G.vs[vid]["journalname"]]
        chosen.append(vid)
        # print(f'{G.vs[vid]["journalname"]} com initial={G.vs[vid]["initial"]} fixed={G.vs[vid]["fixed"]} tem min={min_pesos[vid]} e max={max_pesos[vid]}')

        # adjacency_list = index[adj_mat[vid, :] > 0]
        # print(f'vid:{vid}')
        # for v2 in adjacency_list:
        #     print(f'\tvid:{v2} w:{adj_mat[vid, v2]}')
        # print(50*'-')
    else:
        G.vs[vid]["initial"] = -1
        G.vs[vid]["fixed"] = False
print(50*'*')        
with open('../data/index_to_journalname' + in_name + '.pickle', 'wb') as handle:
    pickle.dump(index_to_journalname, handle, protocol=2)

if do_mds:
    # Generating 
    if nan_sub:
        adj_mat = np.where(adj_mat == 0, np.nan, adj_mat)

    if function == "reciprocal":
        adj_mat = adj_mat + 1e-10
        distance = 1/adj_mat
    elif function == "reverse":
        distance =  np.nanmin(adj_mat) + np.nanmax(adj_mat) - adj_mat
    else:
        raise TypeError(f'Function {function} not defined.')
    del adj_mat
    np.fill_diagonal(distance, 0)
    
    # eps for DBSCAN
    eps = {0: {2: 0.00518, 3: 0.0148, 4: 0.0295, 5: 0.0535, 6: 0.0756, 7: 0.0946, 32: 0.2382, 64: 0.2778, 128: 0.3033},
           1: {2: 0.0048, 3: 0.0141, 4: 0.0291, 5: 0.0524, 6: 0.0736, 7: 0.0933, 32: 0.2382, 64: 0.2778, 128: 0.3033},
           2: {2: 0.00485, 3: 0.0139, 4: 0.028, 5: 0.0475, 6: 0.0676, 7: 0.0877, 32: 0.2382, 64: 0.2778, 128: 0.3033},
           3: {2: 0.00404, 3: 0.0134, 4: 0.0301, 5: 0.0454, 6: 0.0685, 7: 0.089, 32: 0.2382, 64: 0.2778, 128: 0.3033}}
    for w_o in ['d-2']:
        for n_comps in [n_components]:
            embedders = [MDS(ndim=n_comps, weight_option=w_o, itmax=10000)]
            for embedding in embedders:
                # Embedding
                filename = f"../data/distance_embedded/X_transformed_{n_comps}dim_{w_o}weight_{function}function_{nan_sub}nan.npy"
                print(f'Calculando X transormed para {n_comps} dimensoes')
                if os.path.exists(filename):
                    with open(filename, "rb") as f:
                        X_transformed = np.load(f)
                else:
                    try:
                        mds_model = embedding.fit(distance) # shape = journals x n_components
                        X_transformed = mds_model['conf']
                        with open(filename, "wb") as f:
                            np.save(f, X_transformed)
                        print(f'Stress = {mds_model["stress"]} com {n_comps} componentes')
                        del mds_model
                    except:
                        X_transformed = 0
                        print(f'nao deu pra {n_comps} dimensoes')
                        continue
                print(f'X transormed para {n_comps} dimensoes calculado')

                # DBSCAN
                # dbscan_c = DBSCAN(eps=eps[w_o][n_comps])
                # dbscan_c.fit(X_transformed)
                # print(f'MDS DBSCAN weights={w_o} eps={eps[w_o][n_comps]} n_components = {n_comps}')
                # VC = show_communities_length(dbscan_c.labels_)

                # file_out = open(f"../data/original_output/dbscan{test_name}_{n_comps}dim_{w_o}weights_{in_name}.txt", "w")
                # info(file_out, VC, index_to_journalname, lideres, lista_iniciais, G, X_transformed, metric='euclidean', only_ground_truth=False, only_labeled=True)
                # file_out.close()

                for n_clus in [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]:
                    # Clustering
                    # GMM
                    print(f'MDS GMM weights={w_o} n_clusters={n_clus} n_components = {n_comps}')
                    try:
                        gmm_c = GaussianMixture(n_components=n_clus, random_state=RANDOM_STATE)
                        gmm_c.fit(X_transformed)
                        print(f'{"converged" if gmm_c.converged_ else "did not converge"} with {gmm_c.n_iter_} iterations')
                        VC = show_communities_length(gmm_c.predict(X_transformed))

                        # file_out = open(f'../data/gmm_{function}_{mode}_d{n_comps}_c{n_clus}_weights{w_o}.txt', "w")
                        file_out = open(f'../data/trash.txt', "w")
                        info(file_out, VC, index_to_journalname, lideres, lista_iniciais, G, X_transformed, metric='euclidean', only_ground_truth=False, only_labeled=True)
                        file_out.close()
                    except:
                        _ = 1
                        
                    # K-means
                    print(f'MDS KMeans weights={w_o} n_clusters={n_clus} n_components = {n_comps}')
                    try:
                        k_means = KMeans(n_clusters=n_clus, algorithm='elkan', random_state=RANDOM_STATE)
                        k_means.fit(X_transformed)
                        VC = show_communities_length(k_means.labels_)

                        file_out = open(f'../data/trash.txt', "w")
                        info(file_out, VC, index_to_journalname, lideres, lista_iniciais, G, X_transformed, metric='euclidean', only_ground_truth=False, only_labeled=True)
                        file_out.close()
                    except:
                        _ = 1

                    # plt.scatter(np.clip(X_transformed[:, 0], -3, 0.5), np.clip(X_transformed[:, 1], -1, 8), c=k_means.labels_, s=0.4)
                    # plt.savefig(f'../data/kmeans_{function}_{mode}_d{n_comps}_c{n_clus}_weights{w_o}.pdf')

                    # VBGMM
                    print(f'MDS VBGMM weights={w_o} n_clusters={n_clus} n_components = {n_comps}')
                    try:
                        vbgmm_c = BayesianGaussianMixture(n_components=n_clus, weight_concentration_prior=0.01, max_iter=1700, random_state=RANDOM_STATE)
                        vbgmm_c.fit(X_transformed)
                        print(f'{"converged" if vbgmm_c.converged_ else "did not converge"} with {vbgmm_c.n_iter_} iterations')
                        vbgmm_labels = vbgmm_c.predict(X_transformed)
                        VC = show_communities_length(vbgmm_labels)

                        file_out = open(f'../data/trash.txt', "w")
                        info(file_out, VC, index_to_journalname, lideres, lista_iniciais, G, X_transformed, metric='euclidean', only_ground_truth=False, only_labeled=True)
                        file_out.close()
                    except:
                        _ = 1
                del X_transformed

elif opcao_grafo != 2:
    if function != 'agglomerative':
        model = ClusterRec(function=function, threshold=0, times=TIMES).fit(G)
        
        # para o finder.py
        with open('../data/children_multilevel_' + mode + in_name + '.pickle', 'wb') as f:
            pickle.dump(model.children_, f, protocol=2)
        
        file_out = open(f"../data/{function}{test_name}{in_name}.txt", "w")
        distance =  np.nanmin(adj_mat) + np.nanmax(adj_mat) - adj_mat
        np.fill_diagonal(distance, 0)
        labels = info(file_out, model.VC, index_to_journalname, lideres, lista_iniciais, G, distance)
        del model
        file_out.close()

    elif function == 'agglomerative':
        # Processing the authors sets of each journal
        with open('../data/journals_dict' + in_name + '.pickle', 'rb') as handle:
            journals = pickle.load(handle)

        new_journals = dict(journals)
        for key in journals:
            new_journals[key].pop('journal_name', None)
            new_journals[key].pop('journal_name_rough', None)
            if(new_journals[key].__len__() == 0):
                new_journals.pop(key, None)
        journals = new_journals
        
        list_of_authors = []
        journal_ind = {}
        for index, journal in enumerate(journals):
            list_of_authors.append(journals[journal])
            journal_ind[journal] = index
            if 'journal_name' in list_of_authors[-1] or 'journal_name_rough' in list_of_authors[-1]:
                print(f'{journal} errado')

        authors_sets = []
        for authors in list_of_authors:
            new_set = set()
            for author in authors:
                new_set.add(author)
            authors_sets.append(new_set)

        if TIMES == 0:
            TIMES = np.inf
        
        distance =  np.nanmin(adj_mat) + np.nanmax(adj_mat) - adj_mat
        np.fill_diagonal(distance, 0)
        model = Agglomerative(mode=mode).fit(adj_mat=adj_mat, authors_sets=authors_sets, metrics_it=1, max_iter=TIMES, ground_truth=index_to_ground_truth, debug=True)

        # para o finder.py
        with open('../data/children_agglomerative_' + mode + in_name + '.npy', 'wb') as f:
            np.save(f, model.children_)

        # show best metrics
        it = np.argmax(model.metrics_, axis=0)
        best_metrics = ["Adjusted Rand Score", 
                        "Normalized Mutual Information",
                        "Homogeneity",
                        "Completeness",
                        "V-measure",
                        "Fowlkes-Mallows"]
        for m_ind, m_name in enumerate(best_metrics):
            print(f'Best {m_name}: {model.metrics_[it[m_ind], m_ind]} at iteration {it[m_ind]+1}')
        
        # plot dendogram
        if TIMES == np.inf:
            p = int(adj_mat.shape[0] - it[0])
            min_d, max_d = np.min(model.distances_[-p:]), np.max(model.distances_[-p:])
            model.distances_[-p:] = (model.distances_[-p:] - min_d) / (max_d - min_d)
            matplotlib.rcParams['lines.linewidth'] = 0.05
            plt.title(f'Hierarchical Clustering Dendrogram. p = {p} mode = {mode}')
            # plot the top three levels of the dendrogram
            model.plot_dendrogram(truncate_mode='lastp', p=p, distance_sort=True)
            plt.xlabel("Number of points in node (or index of point if no parenthesis).")
            plt.savefig(f'../data/dendogram{in_name}_p{p}_mode{mode}.pdf')
        else:
            # save in a formatted file
            VC = show_communities_length(model.labels_)
            labels_agg = [0]*adj_mat.shape[0]
            for comm_ind, tuple in enumerate(VC):
                ind, comm = tuple[0], tuple[1]
                for v in comm:
                    labels_agg[v] = comm_ind

            file_out = open(f"../data/{function}{test_name}{in_name}.txt", "w")
            info(file_out, VC, index_to_journalname, lideres, lista_iniciais, G, distance)
            file_out.close()

    print(50*'-')
    print(f'Fim {function}')
    print(50*'-')
