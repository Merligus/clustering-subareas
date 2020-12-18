import matplotlib.pyplot as plt
from igraph import *
import numpy as np
import xml.etree.ElementTree as ET
import pickle

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
    print("VC", G.modularity(VC, weights=G.es["commonauthors"]))

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

# 0: nao direcionado
# 1: bidirecionado
# 2: bipartido
opcao_grafo = 0

# Supervisionado ou não supervisionado
supervisionado = False
log_transf = True
do_spanning_tree = False
mode = 'mean' # mean, min, union

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
    seeds_escolhida = 2
    if seeds_escolhida == 1:
        initial = {"ASPLOS": 0,
                    "DAC": 0,
                    "FCCM": 0,
                    "HPCA": 0,
                    "ICCAD": 0,
                    "ISCA": 0,
                    "MICRO": 0,
                    "COLT": 1,
                    "FOCS": 1,
                    "ISSAC": 1,
                    "LICS": 1,
                    "SODA": 1,
                    "STOC": 1,
                    "BIBE": 2,
                    "CSB": 2,
                    "ISMB": 2,
                    "RECOMB": 2,
                    "WABI": 2,
                    "CHES": 3,
                    "EUROCRYPT": 3,
                    "FSE": 3,
                    "ASIACRYPT": 3,
                    "CRYPTO": 3,
                    "TCC": 3,
                    "DEXA": 4,
                    "EDBT": 4,
                    "ER": 4,
                    "ICDT": 4,
                    "PODS": 4,
                    "SIGMOD Conference": 4,
                    "VLDB": 4,
                    "CIKM": 5,
                    "ECML": 5,
                    "ICDE": 5,
                    "ICDM": 5,
                    "ICML": 5,
                    "KDD": 5,
                    "PAKDD": 5,
                    "Euro-Par": 6,
                    "ICDCS": 6,
                    "ICPP": 6,
                    "IPDPS": 6,
                    "PACT": 6,
                    "PODC": 6,
                    "PPoPP": 6,
                    "CGI": 7,
                    "CVPR": 7,
                    "ECCV": 7,
                    "ICCV": 7,
                    "SI3D": 7,
                    "SIGGRAPH": 7,
                    "ICNP": 8,
                    "INFOCOM": 8,
                    "LCN": 8,
                    "Mobihoc": 8,
                    "SIGCOMM": 8,
                    "ACL": 9,
                    "EACL": 9,
                    "ECIR": 9,
                    "NAACL": 9,
                    "SIGIR": 9,
                    "SPIRE": 9,
                    "TREC": 9,
                    "APLAS": 10,
                    "CP": 10,
                    "ICFP": 10,
                    "ICLP": 10,
                    "OOPSLA": 10,
                    "PLDI": 10,
                    "POPL": 10,
                    "ASE": 11,
                    "CAV": 11,
                    "FM": 11,
                    "FME": 11,
                    "SIGSOFT FSE": 11,
                    "ICSE": 11,
                    "PEPM": 11,
                    "TACAS": 11,
                    "CCS": 12,
                    "CSFW": 12,
                    "ESORICS": 12,
                    "NDSS": 12,
                    "EC-Web": 13,
                    "ICWE": 13,
                    "ISWC": 13,
                    "WISE": 13,
                    "WWW": 13}
    else:
        initial = {"Artif. Intell.": 0,
            "J. Artif. Intell. Res. (JAIR)": 0,
            "J. Autom. Reasoning": 0,
            "AAAI": 0,
            "IJCAI": 0,
            "BMC Bioinform.": 1,
            "Bioinformatics [ISMB/ECCB]": 1,
            "Bioinformatics [ISMB]": 1,
            "Bioinformatics": 1,
            "PLoS Computational Biology": 1,
            "RECOMB": 1,
            "IEEE/ACM Trans. Comput. Biology Bioinform.": 1,
            "IEEE/ACM Trans. Netw.": 2,
            "IEEE Trans. Communications": 2,
            "MOBICOM": 2,
            "SIGCOMM": 2,
            "INFOCOM": 2,
            "OOPS Messenger": 3,
            "POPL": 3,
            "SIGPLAN Notices": 3,
            "ACM Trans. Program. Lang. Syst.": 3,
            "CGO": 3,
            "Int. J. Comput. Their Appl.": 4,
            "IEEE Micro": 4,
            "DAC": 4,
            "ASPLOS": 4,
            "IEEE Trans. on CAD of Integrated Circuits and Systems": 4,
            "The Journal of Supercomputing": 4,
            "ACM Trans. Graph.": 5,
            "IEEE Computer Graphics and Applications": 5,
            "IEEE Trans. Vis. Comput. Graph.": 5,
            "IEEE Visualization": 5,
            "SIGGRAPH": 5,
            "ACM Trans. Database Syst.": 6,
            "VLDB J.": 6,
            "PODS": 6,
            "SIGMOD Conference": 6,
            "J. Database Manag.": 6,
            "IEEE Trans. Parallel Distrib. Syst.": 7,
            "J. Parallel Distributed Comput.": 7,
            "PODC": 7,
            "ICDCS": 7,
            "ACM Trans. Comput. Hum. Interact.": 8,
            "Int. J. Man Mach. Stud.": 8,
            "User Model. User-Adapt. Interact.": 8,
            "CHI": 8,
            "CSCW": 8,
            "Int. J. Comput. Vis.": 9,
            "IEEE Trans. Image Process.": 9,
            "CVPR": 9,
            "ICIP": 9,
            "Mach. Learn.": 10,
            "J. Mach. Learn. Res.": 10,
            "Neural Computation": 10,
            "NIPS": 10,
            "ICML": 10,
            "Inf. Syst. Res.": 11,
            "Management Science": 11,
            "J. Manag. Inf. Syst.": 11,
            "Eur. J. Inf. Syst.": 11,
            "MIS Q.": 11,
            "Multimedia Syst.": 12,
            "IEEE Trans. Multimedia": 12,
            "IEEE Multim.": 12,
            "ACM Multimedia": 12,
            "ICME": 12,
            "Math. Program.": 13,
            "SIAM Journal on Optimization": 13,
            "Oper. Res.": 13,
            "INFORMS J. Comput.": 13,
            "Comput. Oper. Res.": 13,
            "Discret. Appl. Math.": 13,
            "ACM Trans. Inf. Syst. Secur.": 14,
            "J. Comput. Secur.": 14,
            "IEEE Secur. Priv.": 14,
            "IEEE Symposium on Security and Privacy": 14,
            "USENIX Security Symposium": 14,
            "ACM Conference on Computer and Communications Security": 14,
            "IEEE Trans. Software Eng.": 15,
            "ACM Trans. Softw. Eng. Methodol.": 15,
            "ICSE": 15,
            "Empirical Software Engineering": 15,
            "TACAS": 15,
            "J. ACM": 16,
            "SIAM J. Comput.": 16,
            "STOC": 16,
            "FOCS": 16,
            "SODA": 16}

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
        VD1 = G.community_fastgreedy(G.es["commonauthors"])
        VC1 = VD1.as_clustering()
        file_out = open(f"../data/Fastgreedy{test_name}.txt", "w")
        info(file_out, VC1, d, lideres, lista_iniciais)
        f.close()

        print(50*'-')
        print('Fim Fastgreedy')
        print(50*'-')

        VC2 = G.community_infomap(edge_weights=G.es["commonauthors"])
        file_out = open(f"../data/Infomap{test_name}.txt", "w")
        info(file_out, VC2, d, lideres, lista_iniciais)
        f.close()

        print(50*'-')
        print('Fim Infomap')
        print(50*'-')

        VC3 = G.community_leading_eigenvector(weights=G.es["commonauthors"])
        file_out = open(f"../data/Leading_eigenvector{test_name}.txt", "w")
        info(file_out, VC3, d, lideres, lista_iniciais)
        f.close()

        print(50*'-')
        print('Fim Leading_eigenvector')
        print(50*'-')

        VC5 = G.community_multilevel(weights=G.es["commonauthors"])
        file_out = open(f"../data/Multilevel{test_name}.txt", 'w')
        info(file_out, VC5, d, lideres, lista_iniciais)
        f.close()

        print(50*'-')
        print('Fim Multilevel')
        print(50*'-')

        # # demora muito
        # VD7 = G.community_edge_betweenness(directed=False, weights=G.es["commonauthors"])
        # VC7 = VD7.as_clustering()
        # file_out = open(f"../data/Edge_betweenness{test_name}.txt", 'w')
        # info(file_out, VC7, d, lideres, lista_iniciais)
        # f.close()

        # # nao funciona com grafo nao conexo
        # VC8 = G.community_spinglass(weights=G.es["commonauthors"])
        # print("VC8", modularity(VC8, G))

        VD9 = G.community_walktrap(weights=G.es["commonauthors"])
        VC9 = VD9.as_clustering()
        file_out = open(f"../data/Walktrap{test_name}.txt", 'w')
        info(file_out, VC9, d, lideres, lista_iniciais)
        f.close()

        print(50*'-')
        print('Fim Walktrap')
        print(50*'-')

        VC6 = G.community_optimal_modularity(G.es["commonauthors"])
        file_out = open(f"../data/Optimal_modularity{test_name}.txt", 'w')
        info(file_out, VC6, d, lideres, lista_iniciais)
        f.close()

        print(50*'-')
        print('Fim Optimal_modularity')
        print(50*'-')
