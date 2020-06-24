from igraph import *
import numpy as np
import xml.etree.ElementTree as ET
import pickle

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



# 0: nao direcionado
# 1: bidirecionado
# 2: bipartido
opcao_grafo = 2

# Gerador no arquivo teste?
test = True
test_name = ""
if test:
    test_name = "test"

# 0: nao direcionado
# 1: bidirecionado
# 2: bipartido
if opcao_grafo == 0:
    G = load("G:\\Mestrado\\BD\\data\\graph_nao_direcionado" + test_name + ".gml")
elif opcao_grafo == 1: 
    G = load("G:\\Mestrado\\BD\\data\\graph_direcionado" + test_name + "2.gml")
else:
    G = load("G:\\Mestrado\\BD\\data\\graph_bipartido" + test_name + ".gml")
    # print(summary(G))
    # V = len(G.vs["name"])
    # for vid in range(V):
    #     print(vid, G.vs[vid]["name"], G.vs[vid]["isjournal"])
    
    # edges = G.get_edgelist()
    # for eid, e in enumerate(edges):
    #     print(e[0], "--", e[1], G.es[eid]["publications"])
    exit()

# VD1 = G.community_fastgreedy(G.es["commonauthors"])
# print(VD1) # junta 2 ANSI
# VC1 = VD1.as_clustering()
# print("VC1", modularity(VC1, G)) #, G.modularity(VC1, weights=G.es["commonauthors"]))
VC2 = G.community_infomap(edge_weights=G.es["commonauthors"])
# print(VC2) # junta 2 ANSI e os 2 LILOG
print("VC2", modularity(VC2, G)) #, G.modularity(VC2, weights=G.es["commonauthors"]))
# VC3 = G.community_leading_eigenvector(weights=G.es["commonauthors"])
# print(VC3) # junta 2 ANSI e os 2 LILOG
# print("VC3", modularity(VC3, G)) #, G.modularity(VC3, weights=G.es["commonauthors"]))
VC4 = G.community_label_propagation(weights=G.es["commonauthors"])
# print(VC4) # junta 2 ANSI e os 2 LILOG
print("VC4", modularity(VC4, G)) #, G.modularity(VC4, weights=G.es["commonauthors"]))
# VC5 = G.community_multilevel(weights=G.es["commonauthors"])
# print(VC5) # junta 2 ANSI e os 2 LILOG
# print("VC5", modularity(VC5, G)) #, G.modularity(VC5, weights=G.es["commonauthors"]))
# VC6 = G.community_optimal_modularity(G.es["commonauthors"])
# print(VC6) # junta 2 ANSI e os 2 LILOG
# print("VC6", modularity(VC6, G)) #, G.modularity(VC6, weights=G.es["commonauthors"]))

# demora muito
# VD7 = G.community_edge_betweenness(directed=True, weights=G.es["commonauthors"])
# print(VD7)
# VC7 = VD7.as_clustering()
# print("VC7", modularity(VC7, G))

# nao funciona com grafo nao conexo
# VC8 = G.community_spinglass(weights=G.es["commonauthors"])
# print("VC8", modularity(VC8, G))

VD9 = G.community_walktrap(weights=G.es["commonauthors"])
VC9 = VD9.as_clustering()
# print(VC9) # junta 2 ANSI e os 2 LILOG
print("VC9", modularity(VC9, G)) #, G.modularity(VC9, weights=G.es["commonauthors"]))
# VC10 = G.community_leiden(objective_function="modularity", weights=G.es["commonauthors"])
# print(VC10) # junta 2 ANSI e os 2 LILOG
# print("VC10", modularity(VC10, G)) #, G.modularity(VC10, weights=G.es["commonauthors"]))

# Nao direcionado
# VC1 0.27147790343889827
# VC2 0.0022660723269686714
# VC3 0.2483238111398965
# VC4 0.0008217862837820856
# VC5 0.2826992759701107
# VC6
# VC7 -
# VC8 -
# VC9 0.2539386929040398
# VC10 0.287020059675666

# Direcionado
# VC1 0.21002349508379062
# VC2 -
# VC3 -
# VC4 0.2026324832940766
# VC5 -
# VC6 -
# VC7 -
# VC8 -
# VC9 0.2482383026421148
# VC10 -

# C = D.as_clustering()
# cid = 0
# for community in C:
#     print("Community %d" %(cid))
#     for vid in community:
#         print("\t vid: %d, journal: %s" %(vid, G.vs[vid]["journalname"]))
#     cid += 1
