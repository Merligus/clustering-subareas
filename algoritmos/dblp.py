import matplotlib.pyplot as plt
from igraph import *
import numpy as np
import xml.etree.ElementTree as ET
import pickle
import sys

from sklearn import metrics
from sklearn.manifold import MDS
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS

def cluster_rec(graph, function, threshold, times=3):
    V = len(graph.vs.indegree())
    graph['names'] = [v for v in range(V)]
    graphs = [graph]
    VC = []
    for _ in range(times):
        new_graphs = []
        for g in graphs:
            # Vertex Cluster
            if function == 'fastgreedy':
                vd = g.community_fastgreedy(weights=g.es['commonauthors'])
                vc = vd.as_clustering()
            elif function == 'infomap':
                vc = g.community_infomap(edge_weights=g.es['commonauthors'])
            elif function == 'leading_eigenvector':
                vc = g.community_leading_eigenvector(weights=g.es['commonauthors'])
            elif function == 'multilevel':
                vc = g.community_multilevel(weights=g.es['commonauthors'])
            elif function == 'walktrap':
                vd = g.community_walktrap(weights=g.es['commonauthors'])
                vc = vd.as_clustering()
            elif function == 'optimal_modularity':
                vd = g.community_optimal_modularity(weights=g.es['commonauthors'])
            elif function == 'label_propagation':
                vc = g.community_label_propagation(weights=g.es["commonauthors"], initial=g.vs["initial"], fixed=g.vs["fixed"])
            elif function == 'community_leiden':
                vc = g.community_leiden(weights=g.es["commonauthors"], initial_membership=g.vs["initial"])
            else:
                print('FUNCTION NOT RECOGNIZED')
            # For each community
            for comm in vc:
                # Too big, execute it again
                if len(comm) > threshold:
                    # Create the subgraph of the community
                    sub_g = g.subgraph(comm, implementation='create_from_scratch')
                    # Saving their real names
                    sub_g['names'] = [g['names'][v] for v in comm]
                    # Insert the subgraph in the queue of execution
                    new_graphs.append(sub_g)
                else:
                    # No need to execute it again, just save their real names
                    VC.append([g['names'][v] for v in comm])
        del graphs
        graphs = new_graphs
    for g in graphs:
        VC.append(g['names'])
    return VC

# only_ground_truth=True, silhouette_ground_truth=True para descobrir o valor maximo que silhouette chega com a matriz "distance"
# only_ground_truth=False, silhouette_ground_truth=True para comparar com o valor maximo do silhouette quando only_ground_truth=True
# silhouette_ground_truth=False. Executa o silhouette na matriz inteira
def info(f, VC, d_ind_pra_nome, representantes, iniciais, G, distance, only_ground_truth=True, silhouette_ground_truth=True, metric='precomputed'):
    labels_true = []
    labels_pred = []
    labels = [0]*G.vs.indegree().__len__()
    initials = []
    initial_labels_pred = []
    for comm_ind, comm in enumerate(VC):
        f.write("{0}\n".format(comm_ind))
        lista_de_jornais = []
        for v in comm:
            if G.vs[v]["fixed"]:
                initials.append(v)
                if only_ground_truth:
                    initial_labels_pred.append(G.vs[v]["initial"])
                else:
                    initial_labels_pred.append(comm_ind)
            labels[v] = comm_ind
            try:
                f.write("\t{1}:{0}\n".format(d_ind_pra_nome[v], comm_ind))
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
    if silhouette_ground_truth: # Com ground truth
        # Apenas pegando as distancias dos jornais do ground truth
        if metric == 'precomputed':
            distance = distance[initials,:][:,initials]
        else:
            distance = distance[initials,:][:,:]
        labels_silhouette = initial_labels_pred
    else: # Sem ground truth
        labels_silhouette = labels
    try:
        print(f'Silhouette: {metrics.silhouette_score(distance, labels_silhouette, metric=metric):.2f}')
    except:
        print(f'Silhouette: -1')
    return labels

def modularity(C, G):
    weights = G.es["commonauthors"]
    W = sum(weights)
    edge_list = G.get_edgelist()
    a_in = [0]*G.vs.indegree().__len__()
    a_out = [0]*G.vs.indegree().__len__()

    comm_index = 0
    membership = [0]*G.vs.indegree().__len__()
    for comm in C:
        for v in comm:
            membership[v] = comm_index
        comm_index += 1
    
    for eid, v in enumerate(edge_list):
        vi, vj = v
        a_out[vi] += weights[eid]
        a_in[vj] += weights[eid]

    mod = 0.0
    for eid, v in enumerate(edge_list):
        vi, vj = v
        ci = membership[vi]
        cj = membership[vj]
        if ci == cj:
            mod += weights[eid]
            mod -= a_out[vi]*a_in[vj]/W

    return mod/W

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
        VC.append([])
    for v, label in enumerate(labels):
        VC[label].append(v)
    return VC

print('STARTING')

if(len(sys.argv) < 6):
    print("Falta parametros")
    exit()
elif(len(sys.argv) > 6):
    print("Muitos parametros")
    exit()
else:
    log_transf = bool(int(sys.argv[1]))
    mode = sys.argv[2] # mean, min, union
    function = sys.argv[3] # multilevel, fastgreedy,  . . .
    TIMES = int(sys.argv[4])
    n_components = int(sys.argv[5])

RANDOM_STATE = 7

# 0: nao direcionado
# 1: bidirecionado
# 2: bipartido
opcao_grafo = 0

do_spanning_tree = False
do_mds = n_components > 1
cut_p = True

# Gerador no arquivo teste?
test = False
year = 0

test_name = '_' + mode
if test:
    test_name += "_test"
if year > 0:
    test_name += '_' + str(year)
if log_transf:
    test_name += '_' + 'log'
test_name += f'_rec{TIMES}'

np.set_printoptions(threshold=np.inf)

# 0: nao direcionado
# 1: bidirecionado
# 2: bipartido
if opcao_grafo == 0:
    filename = "../data/graph_nao_direcionado" + mode + '.npy'
    with open(filename, "rb") as f:
        adj_mat = np.load(f)
    print('LOADED GRAPH')
    with open('../data/journalname.pickle', 'rb') as handle:
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
print(f'Representantes de cada comunidade = {lideres}')
print(50*'*')

V = len(G.vs.indegree())
d = {}
counter = max(initial.values()) + 1
comunidades_antes = counter*[0]
index = np.arange(V)
for vid in range(V):
    d[vid] = G.vs[vid]["journalname"]
    try:
        G.vs[vid]["initial"] = initial[G.vs[vid]["journalname"]]
        G.vs[vid]["fixed"] = True
        comunidades_antes[initial[G.vs[vid]["journalname"]]] += 1
        print(f'{G.vs[vid]["journalname"]} com initial={G.vs[vid]["initial"]} fixed={G.vs[vid]["fixed"]} tem min={min_pesos[vid]} e max={max_pesos[vid]}')

        # adjacency_list = index[adj_mat[vid, :] > 0]
        # print(f'vid:{vid}')
        # for v2 in adjacency_list:
        #     print(f'\tvid:{v2} w:{adj_mat[vid, v2]}')
        # print(50*'-')
    except:
        G.vs[vid]["initial"] = counter
        G.vs[vid]["fixed"] = False
print(50*'*')

if do_mds:
    # Generating 
    # adj_mat = adj_mat + 1e-10
    # distance = 1/adj_mat
    distance = adj_mat.max() - adj_mat
    np.fill_diagonal(distance, 0)
    
    # eps for DBSCAN
    eps = {0: {32: 0.2382, 64: 0.2778, 128: 0.3033},
           1: {32: 0.2382, 64: 0.2778, 128: 0.3033},
           2: {32: 0.2382, 64: 0.2778, 128: 0.3033},
           3: {32: 0.2382, 64: 0.2778, 128: 0.3033}}
    vbgmm_d = {}
    weights_mode = {0: 'normal', 1: '1or0', 2: 'd-1', 3: 'd-2'}
    for w_o in [0, 1, 2, 3]:
        vbgmm_d[w_o] = {}
        for n_comps in [32, 64, 128]:
            embedders = [MDS(n_components=n_comps, dissimilarity='precomputed', metric=True, random_state=RANDOM_STATE, weight_option=w_o)] # TSNE(n_components=n_comps, metric='precomputed', random_state=RANDOM_STATE)]
            for embedding in embedders:
                # Embedding
                # X_transformed = embedding.fit_transform(distance) # shape = journals x n_components
                filename = f"../data/distance_embedded/X_transformed_{n_comps}dim_{weights_mode[w_o]}weight.npy"
                with open(filename, "rb") as f:
                    X_transformed = np.load(f)
                # with open(filename, "wb") as f:
                #     np.save(f, X_transformed)
                # print(f'Stress = {embedding.stress_} com {n_comps} componentes')

                # DBSCAN
                dbscan_c = DBSCAN(eps=eps[w_o][n_comps])
                dbscan_c.fit(X_transformed)
                print(f'MDS DBSCAN eps={eps[w_o][n_comps]} n_components = {n_comps}')
                VC = show_communities_length(dbscan_c.labels_)

                file_out = open(f"../data/trash.txt", "w")
                info(file_out, VC, d, lideres, lista_iniciais, G, X_transformed, metric='euclidean')
                file_out.close()

    #             for n_clus in [20, 40]:
    #                 # Clustering
    #                 # K-means
    #                 k_means = KMeans(n_clusters=n_clus, algorithm='elkan', random_state=RANDOM_STATE)
    #                 k_means.fit(X_transformed)
    #                 print(f'MDS KMeans n_clusters={n_clus} n_components = {n_comps}')
    #                 VC = show_communities_length(k_means.labels_)

    #                 file_out = open(f"../data/trash.txt", "w")
    #                 info(file_out, VC, d, lideres, lista_iniciais, G, X_transformed, metric='euclidean')
    #                 file_out.close()

    #                 # GMM
    #                 gmm_c = GaussianMixture(n_components=n_clus, random_state=RANDOM_STATE)
    #                 gmm_c.fit(X_transformed)
    #                 print(f'MDS GMM n_clusters={n_clus} n_components = {n_comps} {"converged" if gmm_c.converged_ else "did not converge"}')
    #                 VC = show_communities_length(gmm_c.predict(X_transformed))

    #                 file_out = open(f"../data/trash.txt", "w")
    #                 info(file_out, VC, d, lideres, lista_iniciais, G, X_transformed, metric='euclidean')
    #                 file_out.close()

    #                 # VBGMM
    #                 vbgmm_c = BayesianGaussianMixture(n_components=n_clus, weight_concentration_prior=0.01, max_iter=1700, random_state=RANDOM_STATE)
    #                 vbgmm_c.fit(X_transformed)
    #                 print(f'MDS VBGMM n_clusters={n_clus} n_components = {n_comps} {"converged" if vbgmm_c.converged_ else "did not converge"}')
    #                 vbgmm_labels = vbgmm_c.predict(X_transformed)
    #                 VC = show_communities_length(vbgmm_labels)

    #                 # file_out = open(f"../data/VBGMM_{n_clus}clusters_{n_comps}dim_{weights_mode[w_o]}weights.txt", "w")
    #                 file_out = open(f"../data/trash.txt", "w")
    #                 info(file_out, VC, d, lideres, lista_iniciais, G, X_transformed, metric='euclidean')
    #                 file_out.close()

    #                 # add the classified labels in order to compare
    #                 vbgmm_d[w_o][n_comps] = vbgmm_labels

    # # comparing the results
    # for w_o1 in vbgmm_d:
    #     for n_comps1 in vbgmm_d[w_o1]:
    #         for w_o2 in vbgmm_d:
    #             for n_comps2 in vbgmm_d[w_o2]:
    #                 if len(vbgmm_d[w_o1][n_comps1]) == len(vbgmm_d[w_o2][n_comps2]) and (w_o1 != w_o2 or n_comps1 != n_comps2):
    #                     print(50*'*')
    #                     print(f'{weights_mode[w_o1]}weights {n_comps1}dim vs {weights_mode[w_o2]}weights {n_comps2}dim')
    #                     print(50*'*')
    #                     print(f'Adjusted Rand index: {metrics.adjusted_rand_score(vbgmm_d[w_o1][n_comps1], vbgmm_d[w_o2][n_comps2]):.2f}')
    #                     print(f'Adjusted Mutual Information: {metrics.adjusted_mutual_info_score(vbgmm_d[w_o1][n_comps1], vbgmm_d[w_o2][n_comps2]):.2f}')
    #                     print(f'Homogeneity: {metrics.homogeneity_score(vbgmm_d[w_o1][n_comps1], vbgmm_d[w_o2][n_comps2]):.2%}')
    #                     print(f'Completeness: {metrics.completeness_score(vbgmm_d[w_o1][n_comps1], vbgmm_d[w_o2][n_comps2]):.2%}')
    #                     print(f'V-measure: {metrics.v_measure_score(vbgmm_d[w_o1][n_comps1], vbgmm_d[w_o2][n_comps2]):.2%}')
    #                     print(f'Fowlkes-Mallows: {metrics.fowlkes_mallows_score(vbgmm_d[w_o1][n_comps1], vbgmm_d[w_o2][n_comps2]):.2%}')
    #                     print(50*'*')
    #         vbgmm_d[w_o1][n_comps1] = []
                
    # Testing AgglomerativeClustering
    # for n_clus in [300, 500]:
    #     for link in ['average']:
    #         clustering_agglomerative = AgglomerativeClustering(affinity='precomputed',
    #                                                            linkage=link,
    #                                                            # compute_full_tree=True,
    #                                                            n_clusters=n_clus,
    #                                                            # distance_threshold=10**(-e_threshold)
    #                                                            ).fit(distance)
    #         print(f'n_cluster = {n_clus}, link = {link}')
    #         d = {}
    #         for label in clustering_agglomerative.labels_:
    #             if label in d:
    #                 d[label] += 1
    #             else:
    #                 d[label] = 1
    #         print(d)
    #         # print([len(list(group)) for key, group in groupby(clustering_agglomerative.labels_)])
    #         print(clustering_agglomerative.labels_)
    #         file_out = open(f"../data/agglomerative_link_{link}_n_clusters_{n_clus}.txt", "w")
    #         for label in set(clustering_agglomerative.labels_):
    #             file_out.write(f'{label}\n')
    #             for journal, comm_id in enumerate(clustering_agglomerative.labels_):
    #                 if comm_id == label:
    #                     file_out.write(f'\t{comm_id}:{G.vs[journal]["journalname"]}\n')
    #         file_out.close()

    # clustering_affinity = AffinityPropagation(affinity='precomputed', convergence_iter=100, random_state=RANDOM_STATE).fit(distance)
    # print(clustering_affinity.labels_)

    # clustering_spectral_clustering = SpectralClustering(affinity='precomputed', random_state=RANDOM_STATE).fit(distance)
    # print(clustering_spectral_clustering.labels_)

    # clustering_DBSCAN = DBSCAN(metric='precomputed').fit(distance)
    # print(clustering_DBSCAN.labels_)

    # clustering_OPTICS = OPTICS(metric='precomputed').fit(distance)
    # print(clustering_OPTICS.labels_)

elif opcao_grafo != 2:   
    VC = cluster_rec(graph=G, function=function, threshold=50, times=TIMES)
    file_out = open(f"../data/{function}{test_name}.txt", "w")
    labels = info(file_out, VC, d, lideres, lista_iniciais, G)
    print(f'labels: \n{labels}')
    del VC
    file_out.close()

    print(50*'-')
    print(f'Fim {function}')
    print(50*'-')
