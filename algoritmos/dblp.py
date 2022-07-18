from copy import deepcopy
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
def info(f, VC, d_ind_pra_nome, representantes, iniciais, G, distance, only_ground_truth=True, only_labeled=True, metric='precomputed', return_dict=False):
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
    if return_dict:
        labels = {}
        for comm_ind, tuple in enumerate(VC):
            ind, comm = tuple[0], tuple[1]
            for v in comm:
                labels[G.vs[v]["journalname"]] = comm_ind
        return labels
    
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

def load_graph(mode, in_name, cut_p, log_transf, do_spanning_tree):
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
        n_samples = adj_mat.shape[0]
        k = int(n_samples * 0.1)

        adj_mat = np.where(adj_mat == 0, np.inf, adj_mat)
        for _ in range(k):
            ind = np.unravel_index(np.argmin(adj_mat, axis=None), adj_mat.shape)
            adj_mat[ind[0], ind[1]] = 0
            adj_mat[ind[1], ind[0]] = 0
        adj_mat = np.where(adj_mat == np.inf, 0, adj_mat)

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
    return G, adj_mat

def set_initial(G, V, initial):
    index_to_journalname = {}
    journalname_to_index = {}
    index_to_ground_truth = {}
    chosen = []
    for vid in range(V):
        index_to_journalname[vid] = G.vs[vid]["journalname"]
        journalname_to_index[G.vs[vid]["journalname"]] = vid
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
    return index_to_journalname, journalname_to_index, index_to_ground_truth, chosen

print('STARTING')

if(len(sys.argv) < 10):
    print("Falta parametros")
    exit()
elif(len(sys.argv) > 10):
    print("Muitos parametros")
    exit()
else:
    log_transf = bool(int(sys.argv[1]))
    mode = sys.argv[2] # mean, min, union
    function = sys.argv[3] # multilevel, fastgreedy, agglomerative, reciprocal, . . .
    TIMES = int(sys.argv[4])
    n_components = int(sys.argv[5])
    nan_sub = bool(int(sys.argv[6])) # True -> adj_mat[==0] = nan
    only_journals = bool(int(sys.argv[7]))
    cut = float(sys.argv[8])
    year = int(sys.argv[9])
    
if only_journals and cut > 0:
    print("Cut > 0 must have only_journals = False")
    exit()
in_name = ""
if year > 0:
    in_name += '_' + str(year)
if only_journals:
    in_name += '_only_journals'
if cut > 0:
    in_name += f'_cut{cut:.3}'

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

general_journals_impact = False
show_results = False # wrt general_journals_impact
compare_scimago_multilevel = False
compare_only_journals = True

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
    G, adj_mat = load_graph(mode, in_name, cut_p, log_transf, do_spanning_tree)
    V = adj_mat.shape[0]
    
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

initial = {'ai_j': 0,
		   'ai_c': 0,
		   'jair_j': 0,
		   'jar_j': 0,
		   'aaai_c': 0,
		   'ijcai_c': 0,

		   'bmcbi_j': 1,
		   'bioinformatics_j': 1,
		   'jcb_j': 1,
		   'recomb_c': 1,
		   'tcbb_j': 1,

		   'ton_j': 2,
		   'tcom_j': 2,
		   'mobicom_c': 2,
		   'sigcomm_c': 2,
		   'infocom_c': 2,

		   'oopsla_c': 3,
		   'popl_c': 3,
		   'pldi_c': 3,
		   'toplas_j': 3,
		   'cgo_c': 3,

		   'isca_j': 4,
		   'isca_c': 4,
		   'micro_j': 4,
		   'micro_c': 4,
		   'dac_c': 4,
		   'asplos_c': 4,
		   'tcad_j': 4,
		   'tjs_j': 4,

		   'tog_j': 5,
		   'cga_j': 5,
		   'tvcg_j': 5,
		   'siggraph_c': 5,
		   'vis_c': 5,

		   'tods_j': 6,
		   'vldb_j': 6,
		   'vldb_c': 6,
		   'pods_c': 6,
		   'sigmod_j': 6,
		   'sigmod_c': 6,
		   'jdm_j': 6,

		   'tpds_j': 7,
		   'jpdc_j': 7,
		   'podc_c': 7,
		   'icdcs_c': 7,

		   'tochi_j': 8,
		   'ijmms_j': 8,
		   'umuai_j': 8,
		   'chi_c': 8,
		   'cscw_j': 8,
		   'cscw_c': 8,

		   'ijcv_j': 9,
		   'tip_j': 9,
		   'cvpr_c': 9,
		   'icip_c': 9,

		   'ml_j': 10,
		   'ml_c': 10,
		   'jmlr_j': 10,
		   'neco_j': 10,
		   'nips_c': 10,
		   'icml_c': 10,

		   'isr_j': 11,
		   'isr_c': 11,
		   'mansci_j': 11,
		   'jmis_j': 11,
		   'ejis_j': 11,
		   'misq_j': 11,

		   'mms_j': 12,
		   'mms_c': 12,
		   'tmm_j': 12,
		   'ieeemm_j': 12,
		   'mm_c': 12,
		   'icmcs_c': 12,

		   'mp_j': 13,
		   'siamjo_j': 13,
		   'or_j': 13,
		   'or_c': 13,
		   'informs_j': 13,
		   'cor_j': 13,
		   'dam_j': 13,

		   'tissec_j': 14,
		   'jcs_j': 14,
		   'ieeesp_j': 14,
		   'sp_j': 14,
		   'sp_c': 14,
		   'uss_c': 14,
		   'ccs_c': 14,

		   'tse_j': 15,
		   'tosem_j': 15,
		   'icse_c': 15,
		   'ese_j': 15,
		   'ese_c': 15,
		   'tacas_c': 15,
		   
		   'jacm_j': 16,
		   'siamcomp_j': 16,
		   'stoc_c': 16,
		   'focs_c': 16,
		   'soda_c': 16}

fast_journalname = {}
for journal in G.vs["journalname"]:
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

index_to_journalname, journalname_to_index, index_to_ground_truth, chosen = set_initial(G, V, initial)

with open('../data/index_to_journalname_' + mode + in_name + '.pickle', 'wb') as handle:
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
    
    for w_o in ['normal', '1or0', 'd-1', 'd-2']:
        for n_comps in [n_components]:
            embedders = [MDS(ndim=n_comps, weight_option=w_o, verbose=False)]
            for embedding in embedders:
                # Embedding
                filename = f"../data/distance_embedded/X_transformed_{n_comps}dim_{w_o}weight_{function}function_{nan_sub}nan_{mode}{in_name}.npy"
                print(f'Calculando X transormed para {n_comps} dimensoes')
                if os.path.exists(filename):
                    with open(filename, "rb") as f:
                        X_transformed = np.load(f)
                else:
                    # try:
                    mds_model = embedding.fit(distance) # shape = journals x n_components
                    X_transformed = mds_model['conf']
                    with open(filename, "wb") as f:
                        np.save(f, X_transformed)
                    print(f'Stress = {mds_model["stress"]} com {n_comps} componentes')
                    del mds_model
                    # except:
                    #     X_transformed = 0
                    #     print(f'nao deu pra {n_comps} dimensoes')
                    #     continue
                print(f'X transormed para {n_comps} dimensoes calculado')

                for n_clus in [200]:
                    # Clustering
                    # GMM
                    print(f'MDS GMM weights={w_o} n_clusters={n_clus} n_components = {n_comps}')
                    gmm_c = GaussianMixture(n_components=n_clus, random_state=RANDOM_STATE)
                    gmm_c.fit(X_transformed)
                    print(f'{"converged" if gmm_c.converged_ else "did not converge"} with {gmm_c.n_iter_} iterations')
                    VC = show_communities_length(gmm_c.predict(X_transformed))

                    file_out = open(f'../data/gmm_{function}_{mode}_d{n_comps}_c{n_clus}_weights{w_o}.txt', "w")
                    info(file_out, VC, index_to_journalname, lideres, lista_iniciais, G, X_transformed, metric='euclidean', only_ground_truth=False, only_labeled=True)
                    file_out.close()
                        
                    # K-means
                    print(f'MDS KMeans weights={w_o} n_clusters={n_clus} n_components = {n_comps}')
                    k_means = KMeans(n_clusters=n_clus, algorithm='elkan', random_state=RANDOM_STATE)
                    k_means.fit(X_transformed)
                    VC = show_communities_length(k_means.labels_)

                    file_out = open(f'../data/kmeans_{function}_{mode}_d{n_comps}_c{n_clus}_weights{w_o}.txt', "w")
                    info(file_out, VC, index_to_journalname, lideres, lista_iniciais, G, X_transformed, metric='euclidean', only_ground_truth=False, only_labeled=True)
                    file_out.close()

                    # plt.scatter(np.clip(X_transformed[:, 0], -3, 0.5), np.clip(X_transformed[:, 1], -1, 8), c=k_means.labels_, s=0.4)
                    # plt.savefig(f'../data/kmeans_{function}_{mode}_d{n_comps}_c{n_clus}_weights{w_o}.pdf')

                    # VBGMM
                    print(f'MDS VBGMM weights={w_o} n_clusters={n_clus} n_components = {n_comps}')
                    vbgmm_c = BayesianGaussianMixture(n_components=n_clus, weight_concentration_prior=0.01, max_iter=1700, random_state=RANDOM_STATE)
                    vbgmm_c.fit(X_transformed)
                    print(f'{"converged" if vbgmm_c.converged_ else "did not converge"} with {vbgmm_c.n_iter_} iterations')
                    vbgmm_labels = vbgmm_c.predict(X_transformed)
                    VC = show_communities_length(vbgmm_labels)

                    file_out = open(f'../data/vbgmm_{function}_{mode}_d{n_comps}_c{n_clus}_weights{w_o}.txt', "w")
                    info(file_out, VC, index_to_journalname, lideres, lista_iniciais, G, X_transformed, metric='euclidean', only_ground_truth=False, only_labeled=True)
                    file_out.close()
                del X_transformed

elif opcao_grafo != 2:
    if function != 'agglomerative':
        cluster_rec = ClusterRec(function=function, threshold=0, times=TIMES)
        if general_journals_impact:
            G_general = pickle.loads(pickle.dumps(G))
        random.seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        model = cluster_rec.fit(G)
        
        # para o finder.py
        with open('../data/children_multilevel_' + mode + in_name + '.pickle', 'wb') as f:
            pickle.dump(model.children_, f, protocol=2)
        
        distance =  np.nanmin(adj_mat) + np.nanmax(adj_mat) - adj_mat
        np.fill_diagonal(distance, 0)
        if compare_only_journals:            
            if show_results:
                file_out = open(f"../data/oj_vs_conf_{function}{test_name}{in_name}_conf.txt", "w")
            else:
                file_out = None
            d_normal = info(file_out, model.VC, index_to_journalname, lideres, lista_iniciais, G, distance, return_dict=not show_results)
            if show_results:
                file_out.close()
            
            in_name_only_journals = ""
            if year > 0:
                in_name_only_journals += '_' + str(year)
            in_name_only_journals += '_only_journals'
            G_only_journals, adj_mat_only_journals = load_graph(mode, in_name_only_journals, cut_p, log_transf, do_spanning_tree)
            V_only_journals = adj_mat_only_journals.shape[0]
            
            index_to_journalname_oj, journalname_to_index_oj, index_to_ground_truth_oj, chosen_oj = set_initial(G_only_journals, V_only_journals, initial)
            distance_oj =  np.nanmin(adj_mat_only_journals) + np.nanmax(adj_mat_only_journals) - adj_mat_only_journals
            np.fill_diagonal(distance_oj, 0)
            
            random.seed(RANDOM_STATE)
            np.random.seed(RANDOM_STATE)
            model_only_journals = cluster_rec.fit(G_only_journals)
            
            if show_results:
                file_out = open(f"../data/oj_vs_conf_{function}{test_name}{in_name}_only_journals.txt", "w")
            else:
                file_out = None
            d_compare = info(file_out, model_only_journals.VC, index_to_journalname_oj, lideres, lista_iniciais, G_only_journals, distance_oj, return_dict=not show_results)
            if show_results:
                file_out.close()
            
            del model_only_journals
            
            if not show_results:
                labels_normal = []
                labels_compare = []
                for journal_id in d_normal:
                    if journal_id in d_compare:
                        labels_normal.append(d_normal[journal_id])
                        labels_compare.append(d_compare[journal_id])

                print(f'Adjusted Rand index: {metrics.adjusted_rand_score(labels_normal, labels_compare):.2f}')
                print(f'Adjusted Mutual Information: {metrics.adjusted_mutual_info_score(labels_normal, labels_compare):.2f}')
                print(f'Homogeneity: {metrics.homogeneity_score(labels_normal, labels_compare):.2%}')
                print(f'Completeness: {metrics.completeness_score(labels_normal, labels_compare):.2%}')
                print(f'V-measure: {metrics.v_measure_score(labels_normal, labels_compare):.2%}')
                print(f'Fowlkes-Mallows: {metrics.fowlkes_mallows_score(labels_normal, labels_compare):.2%}')
        elif compare_scimago_multilevel:
            d_normal = info(None, model.VC, index_to_journalname, lideres, lista_iniciais, G, distance, return_dict=True)
            labels_normal = []
            labels_compare = []
            for comm_id, file_suffix in enumerate(['AI', 'APP', 'CG', 'CV', 'HCI', 'HW', 'IS', 'NET', 'SP', 'SW', 'TM']):
                with open(f'../data/journal_names_scimago_{file_suffix}.txt') as fr:
                    for line in fr:
                        final_ind = line.find(':')
                        journal_id = line[:final_ind]
                        if journal_id in d_normal:
                            labels_normal.append(d_normal[journal_id])
                            labels_compare.append(comm_id)

            print(f'Adjusted Rand index: {metrics.adjusted_rand_score(labels_normal, labels_compare):.2f}')
            print(f'Adjusted Mutual Information: {metrics.adjusted_mutual_info_score(labels_normal, labels_compare):.2f}')
            print(f'Homogeneity: {metrics.homogeneity_score(labels_normal, labels_compare):.2%}')
            print(f'Completeness: {metrics.completeness_score(labels_normal, labels_compare):.2%}')
            print(f'V-measure: {metrics.v_measure_score(labels_normal, labels_compare):.2%}')
            print(f'Fowlkes-Mallows: {metrics.fowlkes_mallows_score(labels_normal, labels_compare):.2%}')
        elif general_journals_impact:            
            if show_results:
                file_out = open(f"../data/impact_{function}{test_name}{in_name}_normal.txt", "w")
            else:
                file_out = None
            d_normal = info(file_out, model.VC, index_to_journalname, lideres, lista_iniciais, G, distance, return_dict=not show_results)
            if show_results:
                file_out.close()
            
            # taking off general journals
            vertices_to_delete = []
            deleted_vertices_names = set()
            with open('../data/journal_names_scimago_misc.txt') as fr:
                for line in fr:
                    final_ind = line.find(':')
                    journal_id = line[:final_ind]
                    if journal_id in fast_journalname:
                        vertices_to_delete.append(journalname_to_index[journal_id])
                        deleted_vertices_names.add(journal_id)
            es_select = G_general.es.select(_source_in = vertices_to_delete)
            print(50*'*')
            print(50*'*')
            print(f"General venues: {' '.join(deleted_vertices_names)}")
            print(50*'*')
            print(50*'*')
            G_general.delete_edges(es_select)
            
            random.seed(RANDOM_STATE)
            np.random.seed(RANDOM_STATE)
            model_general = cluster_rec.fit(G_general)
            
            if show_results:
                file_out = open(f"../data/impact_{function}{test_name}{in_name}_general.txt", "w")
            else:
                file_out = None
            d_compare = info(file_out, model_general.VC, index_to_journalname, lideres, lista_iniciais, G_general, distance, return_dict=not show_results)
            if show_results:
                file_out.close()
            
            del model_general
            
            if not show_results:
                labels_normal = []
                labels_compare = []
                for journal_id in d_normal:
                    if journal_id not in deleted_vertices_names:
                        labels_normal.append(d_normal[journal_id])
                        labels_compare.append(d_compare[journal_id])
                
                print(f'Adjusted Rand index: {metrics.adjusted_rand_score(labels_normal, labels_compare):.2f}')
                print(f'Adjusted Mutual Information: {metrics.adjusted_mutual_info_score(labels_normal, labels_compare):.2f}')
                print(f'Homogeneity: {metrics.homogeneity_score(labels_normal, labels_compare):.2%}')
                print(f'Completeness: {metrics.completeness_score(labels_normal, labels_compare):.2%}')
                print(f'V-measure: {metrics.v_measure_score(labels_normal, labels_compare):.2%}')
                print(f'Fowlkes-Mallows: {metrics.fowlkes_mallows_score(labels_normal, labels_compare):.2%}')
        else:
            file_out = open(f"../data/{function}{test_name}{in_name}.txt", "w")
            info(file_out, model.VC, index_to_journalname, lideres, lista_iniciais, G, distance)
            file_out.close()
        del model

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
        agglomerative = Agglomerative(mode=mode)
        
        random.seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)
        model = agglomerative.fit(adj_mat=adj_mat, authors_sets=authors_sets, metrics_it=1, max_iter=TIMES, ground_truth=index_to_ground_truth, debug=True)

        # para o finder.py
        with open('../data/children_agglomerative_' + mode + in_name + '.pickle', 'wb') as f:
            pickle.dump(model.children_.tolist(), f, protocol=2)

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
