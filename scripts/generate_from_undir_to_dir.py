from igraph import *
import numpy as np
import xml.etree.ElementTree as ET
import pickle

# Gerador no arquivo teste?
test = False
test_name = ""
if test:
    test_name = "test"

G = load("G:\\Mestrado\\BD\\data\\graph_nao_direcionado" + test_name + "_2015.gml")
print(2)
V = G.vs.indegree().__len__()
edges = []
edge_list = G.get_edgelist()
va = dict()
ea = dict()
va["nauthors"] = G.vs["nauthors"]
va["journalname"] = G.vs["journalname"]
commonauthors = G.es["commonauthors"]
ea["commonauthors"] = []
for e in edge_list:
    v1, v2 = e
    edges.append((v1, v2))
    edges.append((v2, v1))
    eid = G.get_eid(v1, v2)
    w = G.es[eid]["commonauthors"]
    l1 = G.vs[v1]["nauthors"]
    l2 = G.vs[v2]["nauthors"]
    common = round(w * (l1 + l2)/(1.0 + w))
    ea["commonauthors"].append(common/l1)
    ea["commonauthors"].append(common/l2)

G2 = Graph(n=V, edges=edges, vertex_attrs=va, edge_attrs=ea, directed=True)
G2.save("G:\\Mestrado\\BD\\data\\graph_direcionado" + test_name + "_2015.gml")