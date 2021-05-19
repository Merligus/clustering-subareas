import matplotlib.pyplot as plt
from igraph import *
import numpy as np
import xml.etree.ElementTree as ET
import pickle
import sys

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

def info(f, VC, d_ind_pra_nome, representantes, iniciais):
    for comm_ind, comm in enumerate(VC):
        f.write("{0}\n".format(comm_ind))
        lista_de_jornais = []
        for v in comm:
            try:
                f.write("\t{0}\n".format(d_ind_pra_nome[v]))
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
                    print(f'{name} da comunidade {i} foi para a comunidade {comm_ind}')
    # print("VC", G.modularity(VC, weights=G.es["commonauthors"]))

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

print('STARTING')

if(len(sys.argv) < 4):
    print("Falta parametros")
    exit()
elif(len(sys.argv) > 4):
    print("Muitos parametros")
    exit()
else:
    log_transf = bool(int(sys.argv[1]))
    mode = sys.argv[2] # mean, min, union
    function = sys.argv[3] # mean, min, union

# 0: nao direcionado
# 1: bidirecionado
# 2: bipartido
opcao_grafo = 0

# Supervisionado ou não supervisionado
supervisionado = False
do_spanning_tree = False

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

np.set_printoptions(threshold=np.inf)

# 0: nao direcionado
# 1: bidirecionado
# 2: bipartido
if opcao_grafo == 0:
    filename = "../data/graph_nao_direcionado" + mode + '.npy'
    with open(filename, "rb") as f:
        adj_mat = np.load(f)
    print('LOADED GRAPH')
    with open('../data/journalname' + '.pickle', 'rb') as handle:
        journalname = pickle.load(handle)
    print('LOADED JOURNAL NAMES')
    V = adj_mat.shape[0]

    ################## iGraph
    G = Graph()
    G.add_vertices(V)
    edges = []
    weights = []
    v1 = 0
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

if opcao_grafo != 2:
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
    comunidades_indices_a = [i for i in range(counter)]
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
            counter += 1
            G.vs[vid]["fixed"] = False
    print(50*'*')

    # Algoritmos supervisionados
    if supervisionado:
        file_out = open(f"../data/Label_propagation{test_name}.txt", "w")
        VC4 = G.community_label_propagation(weights=G.es["commonauthors"], initial=G.vs["initial"], fixed=G.vs["fixed"])
        comunidades_indices_d = [i for i in range(len(VC4))]
        comunidades_depois = len(VC4)*[0]
        for comm_ind, comm in enumerate(VC4):
            file_out.write("{0}\n".format(comm_ind))
            comunidades_depois[comm_ind] = len(comm)
            lista_de_jornais = []
            for v in comm:
                try:
                    file_out.write("\t{0}\n".format(d[v]))
                    lista_de_jornais.append(d[v])
                except:
                    a = 1
            # Mostrando o número de jornais/conferências em cada comunidade 
            # antes e depois do label propagation
            for i in lideres:
                if lideres[i] in lista_de_jornais:
                    print(f'Comunidade {i} de {lideres[i]} com {comunidades_antes[i]} jornais/confs foi para {len(comm)}')
                    break

        file_out.close()
        print("VC4", G.modularity(VC4, weights=G.es["commonauthors"]))

        plt.bar(comunidades_indices_a, comunidades_antes, label='Antes', color='black')
        plt.legend()
        plt.xlabel('Índice da comunidade')
        plt.ylabel('Número de comunidades')
        plt.title('Comunidades antes do\nLabel Propagation')
        plt.savefig('../data/distribuicao_antes.png')
        plt.clf()

        plt.bar(comunidades_indices_d, comunidades_depois, label='Depois', color='black')
        plt.legend()
        plt.xlabel('Índice da comunidade')
        plt.ylabel('Número de comunidades')
        plt.title('Comunidades depois do\nLabel Propagation')
        plt.savefig('../data/distribuicao_depois.png')

        print(f'Terminou Label Propagation\n\n')

        # Inicio do Leiden
        V = len(G.vs.indegree())
        d = {}
        counter = 17
        for vid in range(V):
            d[vid] = G.vs[vid]["journalname"]
            try:
                G.vs[vid]["initial"] = initial[G.vs[vid]["journalname"]]
                print(G.vs[vid]["journalname"], G.vs[vid]["initial"], G.vs[vid]["fixed"])
            except:
                G.vs[vid]["initial"] = counter
                counter += 1
        
        file_out = open(f"../data/Leiden{test_name}.txt", "w")
        VC10 = G.community_leiden(objective_function="modularity", resolution_parameter=1.16, weights=G.es["commonauthors"], initial_membership=G.vs["initial"])
        for comm_ind, comm in enumerate(VC10):
            file_out.write("{0}\n".format(comm_ind))
            for v in comm:
                try:
                    file_out.write("\t{0}\n".format(d[v]))
                except:
                    a = 1
        file_out.close()
        print("VC10", modularity(VC10, G)) #, G.modularity(VC10, weights=G.es["commonauthors"]))
        print(f'Terminou Leiden\n\n')
    
    # Não-supervisionado
    else:
        VC = cluster_rec(graph=G, function=function, threshold=50)
        file_out = open(f"../data/{function}{test_name}.txt", "w")
        info(file_out, VC, d, lideres, lista_iniciais)
        del VC
        f.close()

        print(50*'-')
        print(f'Fim {function}')
        print(50*'-')

        # fastgreedy infomap leading_eigenvector multilevel walktrap optimal_modularity

        # VC2 = cluster_rec(graph=G, function='infomap', threshold=50)
        # file_out = open(f"../data/Infomap{test_name}.txt", "w")
        # info(file_out, VC2, d, lideres, lista_iniciais)
        # del VC2
        # f.close()

        # print(50*'-')
        # print('Fim Infomap')
        # print(50*'-')

        # VC3 = cluster_rec(graph=G, function='leading_eigenvector', threshold=50)
        # file_out = open(f"../data/Leading_eigenvector{test_name}.txt", "w")
        # info(file_out, VC3, d, lideres, lista_iniciais)
        # del VC3
        # f.close()

        # print(50*'-')
        # print('Fim Leading_eigenvector')
        # print(50*'-')

        # VC5 = cluster_rec(graph=G, function='multilevel', threshold=50)
        # file_out = open(f"../data/Multilevel{test_name}.txt", 'w')
        # info(file_out, VC5, d, lideres, lista_iniciais)
        # del VC5
        # f.close()

        # print(50*'-')
        # print('Fim Multilevel')
        # print(50*'-')

        # # # demora muito
        # # VD7 = G.community_edge_betweenness(directed=False, weights=G.es["commonauthors"])
        # # VC7 = VD7.as_clustering()
        # # file_out = open(f"../data/Edge_betweenness{test_name}.txt", 'w')
        # # info(file_out, VC7, d, lideres, lista_iniciais)
        # # f.close()

        # # # nao funciona com grafo nao conexo
        # # VC8 = G.community_spinglass(weights=G.es["commonauthors"])
        # # print("VC8", modularity(VC8, G))

        # VC9 = cluster_rec(graph=G, function='walktrap', threshold=50)
        # file_out = open(f"../data/Walktrap{test_name}.txt", 'w')
        # info(file_out, VC9, d, lideres, lista_iniciais)
        # del VC9
        # f.close()

        # print(50*'-')
        # print('Fim Walktrap')
        # print(50*'-')

        # VC6 = cluster_rec(graph=G, function='optimal_modularity', threshold=50)
        # file_out = open(f"../data/Optimal_modularity{test_name}.txt", 'w')
        # info(file_out, VC6, d, lideres, lista_iniciais)
        # del VC6
        # f.close()

        # print(50*'-')
        # print('Fim Optimal_modularity')
        # print(50*'-')
