from igraph import *
import numpy as np
import xml.etree.ElementTree as ET
import pickle

def binarySearch(alist, item):
    first = 0
    last = len(alist)-1
    found = False
    while first<=last and not found:
        midpoint = (first + last)//2
        if alist[midpoint] == item:
            found = True
        else:
            if item < alist[midpoint]:
                last = midpoint-1
            else:
                first = midpoint+1
    return found, midpoint

# 0: nao direcionado
# 1: bidirecionado
# 2: bipartido
opcao_grafo = 2

# Gerador no arquivo teste?
test = False
test_name = ""
if test:
    test_name = "test"

with open('G:\\Mestrado\\BD\\data\\journals_dict' + test_name + '.pickle', 'rb') as handle:
    journals = pickle.load(handle)


if opcao_grafo == 2:
    with open('G:\\Mestrado\\BD\\data\\journals_publications_dict' + test_name + '.pickle', 'rb') as handle:
        journals_publications = pickle.load(handle)

    with open('G:\\Mestrado\\BD\\data\\set_of_authors' + test_name + '.pickle', 'rb') as handle:
        set_of_authors = pickle.load(handle)

    new_journals = dict(journals)
    new_journals_publications = dict(journals_publications)
    for key in journals:
        if(journals[key].__len__() == 0):
            del new_journals[key]
            del new_journals_publications[key]
    
    journals = new_journals
    journals_publications = new_journals_publications

    list_of_all_authors = list(set_of_authors)
    dict_of_all_authors = dict()
    for index, author in enumerate(list_of_all_authors):
        dict_of_all_authors[author] = index

    index = 0
    J_len = len(journals)
    shift = J_len
    edges = []
    edges_weights = []
    for journal in journals:
        for a_index, author in enumerate(journals[journal]):
            edges.append((index, shift + dict_of_all_authors[author]))
            edges_weights.append(journals_publications[journal][a_index])
        index += 1

else:
    new_journals = dict(journals)
    for key in journals:
        if(journals[key].__len__() == 0):
            del new_journals[key]

    journals = new_journals

    list_of_authors = []
    index = 0
    journal_ind = {}
    for journal in journals:
        list_of_authors.append(journals[journal])
        list_of_authors[-1].sort()
        journal_ind[journal] = index
        index += 1

if opcao_grafo == 0:
    V = len(journals)
    G = Graph.Full(V)
elif opcao_grafo == 1:
    V = len(journals)
    G = Graph.Full(V, directed=True)
else:
    G = Graph(J_len)
    A_len = len(set_of_authors)
    G.add_vertices(A_len)
    V = J_len + A_len

if opcao_grafo == 2:
    G.add_edges(edges)
    G.es["publications"] = edges_weights

    vid = 0
    for journal in journals:
        G.vs[vid]["name"] = journal
        G.vs[vid]["isjournal"] = 1
        vid += 1

    for author in list_of_all_authors:
        G.vs[vid]["name"] = author
        G.vs[vid]["isjournal"] = 0
        vid += 1
else:
    v1 = 0
    while v1 < V-1:
        v2 = v1 + 1
        while v2 < V:
            common = 0
            l1 = list_of_authors[v1]
            l2 = list_of_authors[v2]
            l1_len = l1.__len__()
            l2_len = l2.__len__()
            if l1_len > l2_len:
                l1, l2 = l2, l1
                l1_len, l2_len = l2_len, l1_len
            for authors1 in l1:
                found, authors2_ind = binarySearch(l2, authors1)
                if found:
                    common += 1
            
            if opcao_grafo == 0:
                eid = G.get_eid(v1, v2)
                if(common == 0):
                    G.delete_edges(eid)
                else:
                    G.es[eid]["commonauthors"] = common/(list_of_authors[v1].__len__() + list_of_authors[v2].__len__() - common)
            else:
                if(common == 0):
                    eid1 = G.get_eid(v1, v2)
                    G.delete_edges(eid1)
                    eid2 = G.get_eid(v2, v1)
                    G.delete_edges(eid2)
                else:
                    eid1 = G.get_eid(v1, v2)
                    G.es[eid1]["commonauthors"] = common/(list_of_authors[v1].__len__())
                    eid2 = G.get_eid(v2, v1)
                    G.es[eid2]["commonauthors"] = common/(list_of_authors[v2].__len__())

            v2 += 1
        v1 += 1

    vid = 0
    for journal in journal_ind:
        ind = journal_ind[journal]
        length = list_of_authors[ind].__len__()
        G.vs[vid]["nauthors"] = length
        G.vs[vid]["journalname"] = journal
        vid += 1

# 0: nao direcionado
# 1: bidirecionado
# 2: bipartido
if opcao_grafo == 0:
    G.save("G:\\Mestrado\\BD\\data\\graph_nao_direcionado" + test_name + ".gml")
elif opcao_grafo == 1: 
    G.save("G:\\Mestrado\\BD\\data\\graph_direcionado" + test_name + ".gml")
else:
    G.save("G:\\Mestrado\\BD\\data\\graph_bipartido" + test_name + ".gml")
