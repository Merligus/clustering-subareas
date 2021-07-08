from igraph import *
import numpy as np
import xml.etree.ElementTree as ET
import pickle
import sys
import html
import time

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
opcao_grafo = 0

# Gerador no arquivo teste?
if (len(sys.argv) < 4):
    print('Falta parametros')
    exit()
elif (len(sys.argv) > 4):
    print('Muitos parametros')
    exit()
else:
    only_journals = bool(int(sys.argv[1]))
    cut = int(sys.argv[2])
    mode = sys.argv[3]
test = False
test_name = ""
year = 0
if test:
    test_name = "_test"
if year > 0:
    test_name += '_' + str(year)
if only_journals:
    test_name += '_only_journals'
if cut > 0:
    test_name += '_cut' + str(cut)

with open('../data/journals_dict' + test_name + '.pickle', 'rb') as handle:
    journals = pickle.load(handle)


if opcao_grafo == 2:
    with open('../data/journals_publications_dict' + mode + '.pickle', 'rb') as handle:
        journals_publications = pickle.load(handle)

    with open('../data/set_of_authors' + mode + '.pickle', 'rb') as handle:
        set_of_authors = pickle.load(handle)

    # file_authors = open('../data/authors.txt', "w", encoding="utf-8")
    # for author in set_of_authors:
    #     try:
    #         file_authors.write(html.escape(author) + "\n")
    #     except:
    #         print(author)

    # file_authors.close()
    # exit()
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
        new_journals[key].pop('journal_name', None)
        new_journals[key].pop('journal_name_rough', None)
        if(new_journals[key].__len__() == 0):
            new_journals.pop(key, None)
            print(f'{key} removido')

    journals = new_journals

    ########### PARA IMPRIMIR OS AUTORES E JORNAIS EM ARQUIVOS TXT
    # file_authors = open('../data/authors.txt', "w", encoding="utf-8")
    # file_journals = open('../data/journals.txt', "w", encoding="utf-8")
    # for j in journals.keys():
    #     try:
    #         file_journals.write(html.escape(j) + "\n")
    #     for a in journals[j].keys():
    #         try:
    #             file_authors.write(html.escape(a) + "\n")

    # file_authors.close()
    # file_journals.close()

    list_of_authors = []
    journal_ind = {}
    for index, journal in enumerate(journals):
        list_of_authors.append(journals[journal])
        journal_ind[journal] = index
        if 'journal_name' in list_of_authors[-1] or 'journal_name_rough' in list_of_authors[-1]:
            print(f'{journal} errado')
        
if opcao_grafo == 0:
    V = len(journals)
elif opcao_grafo == 1:
    V = len(journals)
    G = Graph.Full(V, directed=True)
else:
    G = Graph(J_len)
    A_len = len(set_of_authors)
    G.add_vertices(A_len)
    V = J_len + A_len

adj_mat = np.zeros((V, V))

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
                if l2.get(authors1):
                    common += 1
            if opcao_grafo == 0:
                if common != 0:
                    if mode == 'union':
                        adj_mat[v1, v2] = common/(list_of_authors[v1].__len__() + list_of_authors[v2].__len__() - common)
                    elif mode == 'max':
                        adj_mat[v1, v2] = common/max(list_of_authors[v1].__len__(), list_of_authors[v2].__len__())
                    elif mode == 'mean':
                        denom = (list_of_authors[v1].__len__() + list_of_authors[v2].__len__()) / 2
                        adj_mat[v1, v2] = common/denom
                    elif mode == 'none':
                        adj_mat[v1, v2] = common
                    adj_mat[v2, v1] = adj_mat[v1, v2]
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
        print('{} of {} done'.format(v1, V))
        v1 += 1

    nauthors = [0]*len(list_of_authors)
    journalname = ['-']*len(list_of_authors)
    for journal in journal_ind:
        ind = journal_ind[journal]
        length = list_of_authors[ind].__len__()
        nauthors[ind] = length
        journalname[ind] = journal

del journals
del new_journals
del journal_ind
del list_of_authors

print('Apaguei')

# p = open(filename + '.txt', 'w')
# p.write('V = {}\n'.format(G.vs.indegree().__len__()))
# p.write('vs nauthors\n')
# for vid, nauthors in enumerate(G.vs['nauthors']):
#     p.write('\t{} {}\n'.format(vid, nauthors))
# p.write('vs journalname\n')
# string_type = 'aaa'
# for vid, journalname in enumerate(G.vs['journalname']):
#     try:
#         if type(journal) == type(string_type):
#             string_type = html.escape(journal)
#             p.write('\t{} {}\n'.format(vid, string_type))
#     except:
#         string_type = 'aa'
# p.write('edgelist\n')
# for e in G.get_edgelist():
#     p.write('\t{}-{}\n'.format(e[0], e[1]))
# p.write('es commonauthors\n')
# for eid, commonauthors in enumerate(G.es['commonauthors']):
#     p.write('\t{} {}\n'.format(eid, commonauthors))
# p.close()

# 0: nao direcionado
# 1: bidirecionado
# 2: bipartido

if opcao_grafo == 0:
    with open('../data/graph_nao_direcionado' + mode + test_name + '.npy', 'wb') as f:
        np.save(f, adj_mat)
    with open('../data/nauthors' + test_name + '.pickle', 'wb') as handle:
        pickle.dump(nauthors, handle, protocol=2)
    with open('../data/journalname' + test_name +'.pickle', 'wb') as handle:
        pickle.dump(journalname, handle, protocol=2)
elif opcao_grafo == 1: 
    G.save("../data/graph_direcionado" + mode + ".gml")
else:
    G.save("../data/graph_bipartido" + mode + ".gml")
