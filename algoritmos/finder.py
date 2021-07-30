import argparse
import matplotlib.pyplot as plt
import networkx as nx
from smacof import MDS
import numpy as np
import pickle
from collections import defaultdict
import nltk
import string

class ClusterFinder:

    def __init__(self, children, n_samples, in_set_conf, labels_sets, debug = False):
        self.debug = debug
        self.children = children
        self.n_samples = n_samples
        self.in_set_conf = in_set_conf
        self.labels_sets = labels_sets
        self.nodes = []
        self.v = 0
        self.bad_words = {'cambridge', 'cambridge', 'ca', 'california', 'jose', 'int', 'second', 'third', '4th', '5th', '6th', 
                        '7th', '8th', '9th', '10th', 'san', 'information', 'first', 'va', '3rd', '2nd', 'de', '2010.', 'annual', 
                        'systems', 'selected', 'revised', 'symposium', 'papers', 'acm',
                        'comput', 'computational', 'ieee', 'j.', 'computing', 'international', 'proceeding',
                        'proceedings', 'of', 'to', 'for', 'comp.', 'computer', 'conference', 'workshop',
                        'on', 'and', 'or', 'the', 'that', 'this', 'with', 'good', 'bad', 'show', 'in',
                        'january', 'february', 'march', 'may', 'april', 'june', 'july', 'august', 'october',
                        'september', 'november', 'december', 'journal'}

    def find_cluster(self, initial_it):
        if initial_it == 0:
            for vi in range(self.n_samples):
                self.labels_sets.append({vi})
        
        iteration = len(self.children)
        for index in range(initial_it, len(self.children)):
            # v1, v2 = self.children[index, 0], self.children[index, 1]
            # if v1 >= self.n_samples:
            #     v1 = self.nodes[v1-self.n_samples]
            # if v2 >= self.n_samples:
            #     v2 = self.nodes[v2-self.n_samples]
            # self.labels_sets[v1] = self.labels_sets[v1].union(self.labels_sets[v2])
            
            
            v1 = next(iter(set(self.children[index])))
            if v1 >= self.n_samples:
                v1 = self.nodes[v1-self.n_samples]

            if self.debug:
                print("v1=", v1)
            for vi in set(self.children[index]):
                if vi >= self.n_samples:
                    vi = self.nodes[vi-self.n_samples]
                if vi != v1:
                    if self.debug:
                        print("vi=", vi)
                    self.labels_sets[v1] = self.labels_sets[v1].union(self.labels_sets[vi])
            
            found = True
            if self.debug:
                print(f"labels_sets[{v1}]=", self.labels_sets[v1])
                print("in_set_conf", self.in_set_conf)
            for v in self.in_set_conf:
                if v not in self.labels_sets[v1]:
                    found = False
                    break
                else:
                    if self.debug:
                        print(v, "Aparece")
            
            self.nodes.append(v1)
            # self.labels_sets[v2] = {-1}
            for vi in self.children[index]:
                if vi >= self.n_samples:
                    vi = self.nodes[vi-self.n_samples]
                if vi != v1:
                    self.labels_sets[vi] = {-1}

            if found:
                print("Achou")
                iteration = index
                self.v = v1
                break
            
            if self.debug:
                entrada = input()
        
        return self.v, iteration

    # n = top n
    def show_top(self, sentences, n=10):
        length = len(sentences)
        word_frequence = defaultdict(list)
        frequent = defaultdict(int)
        for i1, s1 in enumerate(sentences[:-1]):
            token1 = nltk.word_tokenize(s1)
            set_token1 = set(token1)
            for i2, s2 in enumerate(sentences[i1+1:]):
                token2 = nltk.word_tokenize(s2)
                set_token2 = set(token2)
                set_r = set_token1.intersection(set_token2)
                for word in set_r:
                    if word not in string.punctuation:
                        if word not in self.bad_words:
                            if not word.isnumeric():
                                frequent[word] += 1
                                word_frequence[word].append(i1)
                                word_frequence[word].append(i1+i2+1)
        top = sorted(frequent.items(), key=lambda item: item[1], reverse=True)[:n]
        for word, _ in top:
            print(f'\t{word} {len(set(word_frequence[word]))/length:.0%}')

parser = argparse.ArgumentParser()
parser.add_argument('-m', action='store', dest='mode', type=str, default='union')
parser.add_argument('-i', action='store', dest='in_name', type=str, default='-')
parser.add_argument('-d', action='store', dest='dir', type=str, default='../data')
parser.add_argument('-f', action='store', dest='function', type=str, default='agglomerative')

parsed = parser.parse_args()
if parsed.in_name != '-':
    parsed.in_name = '_' + parsed.in_name
else:
    parsed.in_name = ''
print(25*'*', 'ARGS', 25*'*')
print(f'Mode: {parsed.mode}')
print(f'In name: {parsed.in_name}')
print(f'Directory: {parsed.dir}')
print(f'Function: {parsed.function}')

filename = f"{parsed.dir}/graph_nao_direcionado" + parsed.mode + parsed.in_name + '.npy'
with open(filename, "rb") as f:
    adj_mat = np.load(f)
distance = np.nanmin(adj_mat) + np.nanmax(adj_mat) - adj_mat

filename = f"{parsed.dir}/children_{parsed.function}_" + parsed.mode + parsed.in_name
if parsed.function == 'agglomerative': # load em python
    with open(filename + '.npy', "rb") as f:
        children = np.load(f)
elif parsed.function == 'multilevel':
    with open(filename + '.pickle', 'rb') as f:
        children = pickle.load(f)
else:
    raise KeyError(f'Função não disponível.')

with open(f'{parsed.dir}/index_to_journalname_{parsed.mode}{parsed.in_name}.pickle', 'rb') as handle:
    index_to_journalname = pickle.load(handle)

with open(f'{parsed.dir}/journals_dict{parsed.in_name}.pickle', 'rb') as handle:
    journals = pickle.load(handle)

index_to_journal_complete_name = {}
for v in index_to_journalname:
    journal = index_to_journalname[v]
    if journal not in journals:
        print(journal)
        continue
    if len(journals[journal]['journal_name']) > 0:
        index_to_journal_complete_name[v] = journal + ': ' + journals[journal]['journal_name']
    elif 'journal_name_rough' in journals[journal]:
        index_to_journal_complete_name[v] = journal + ': ' + journals[journal]['journal_name_rough']
    else:
        print(f"{journal} nome não identificado")
        index_to_journal_complete_name[v] = journal + ': ' + journal.upper()

with open(f'{parsed.dir}/index_to_journal_complete_name{parsed.in_name}.pickle', 'wb') as handle:
    pickle.dump(index_to_journal_complete_name, handle, protocol=2)

with open(f'{parsed.dir}/index_to_journal_complete_name{parsed.in_name}.pickle', 'rb') as handle:
    index_to_journal_complete_name = pickle.load(handle)

with open(f'{parsed.dir}/nauthors{parsed.in_name}.pickle', 'rb') as handle:
    nauthors = pickle.load(handle)

graph_n = 1
while True:
    print(25*'*', 'NOVA PROCURA', 25*'*')
    print("Siglas conferências:")
    entrada = set(input().strip().split())
    if len(entrada) == 0:
        break

    in_set_conf = set()
    s = ''
    for key in index_to_journalname:
        if index_to_journalname[key] in entrada:
            in_set_conf.add(key)
            s += index_to_journalname[key] + ' ' 
    if len(s) == 0:
        print(f'Nenhum identificado')
        continue
    print(f'Identificados: {s}')

    n_samples = len(distance)
    
    print("Procurando...")
    labels_sets = []
    old_cluster = in_set_conf
    cf = ClusterFinder(children, n_samples, in_set_conf, labels_sets)
    c, iteration = cf.find_cluster(0)
    if iteration == len(cf.children):
        print(f"Nenhum cluster achado")
        continue
    cluster = cf.labels_sets[c]
    print(f"Primeiro cluster de tamanho {len(cluster)} juntos após {iteration} iterações")
    
    while True:
        print(25*'*', 'OPÇÕES', 25*'*')
        print("0. Sair")
        print("1. Próxima junção")
        print("2. Mostrar conferências do cluster")
        print("3. Mostrar frequência das palavras do cluster")
        print("4. Mostrar grafo")
        entrada = input().strip()
        if len(entrada) == 0:
            continue
        
        # opcao de sair da atual
        if entrada[0] == '0':
            break
        # opcao de mostrar a proxima juncao do cluster
        elif entrada[0] == '1':
            old_cluster = cluster
            c, iteration = cf.find_cluster(iteration+1)
            if iteration < len(cf.children):
                cluster = cf.labels_sets[c]
                print(f"Cluster de tamanho {len(cluster)} na iteração {iteration}")
            else:
                print(f"Não há próxima iteração")
        # opcao de mostrar o cluster
        elif entrada[0] == '2':
            print(25*'*', 'Velhos', 25*'*')
            i = 0
            for vi in old_cluster:
                print(f"{i}: {index_to_journal_complete_name[vi]}")
                i += 1

            print(25*'*', 'Novos', 25*'*')
            for vi in cluster:
                if vi not in old_cluster:
                    print(f"{i}: {index_to_journal_complete_name[vi]}")
                    i += 1
        # mostra a frequencia das palavras no cluster
        elif entrada[0] == '3':
            sentences = []
            for vi in cluster:
                sentences.append(index_to_journal_complete_name[vi].lower())
            cf.show_top(sentences, n=10)
        # plota o cluster em forma de grafo
        elif entrada[0] == '4':
            g = nx.Graph()

            vertices = []
            for vi in cluster:
                vertices.append(vi)
            
            distance_temp = np.zeros((len(cluster), len(cluster)))
            m = MDS(ndim=2, weight_option="d-2", itmax=10000)

            journalname = {}
            node_size = []
            v1 = 0
            while v1 < len(cluster):
                v2 = v1 + 1
                while v2 < len(cluster):
                    if distance[vertices[v1], vertices[v2]] > 0 and distance[vertices[v1], vertices[v2]] < np.inf:
                        if adj_mat[vertices[v1], vertices[v2]] > 0:
                            g.add_edge(v1, v2, weight=adj_mat[vertices[v1], vertices[v2]])
                        distance_temp[v1, v2] = distance_temp[v2, v1] = distance[vertices[v1], vertices[v2]]
                    v2 += 1
                journalname[v1] = index_to_journalname[vertices[v1]]
                node_size.append(4*np.ceil(nauthors[vertices[v1]]/len(cluster)))
                v1 += 1
            
            mds_model = m.fit(distance_temp) # shape = journals x n_components
            X_transformed = mds_model['conf']

            width = nx.get_edge_attributes(g, 'weight')
            min_w = min(width.values())
            max_w = max(width.values())
            for w in width:
                width[w] = 0.5 + 4*(width[w] - min_w)/(max_w - min_w)
            
            edge_labels = {}
            for v1, v2, w in g.edges.data():
                edge_labels[(v1, v2)] = f"{adj_mat[vertices[v1], vertices[v2]]:.2f}" # w['weight']
                # print(f'{v1}:{journalname[v1]}:{nauthors[v1]}, {v2}:{journalname[v2]}:{nauthors[v2]} = {distance[vertices[v1], vertices[v2]]}:{w["weight"]}')

            fig = plt.figure(figsize=(24,24))
            ax = fig.add_axes([0,0,1,1])
            # pos = nx.spring_layout(g)
            pos = {}
            for vi in range(len(cluster)):
                pos[vi] = X_transformed[vi]
            # print(pos)
            nx.draw_networkx_nodes(g, pos, ax=ax, node_size=node_size, node_color="#A8C1FB")
            nx.draw_networkx_labels(g, pos, ax=ax, labels=journalname, font_color="#DF0000", font_size=22)
            nx.draw_networkx_edges(g, pos, ax=ax, width=list(width.values()))
            nx.draw_networkx_edge_labels(g, pos, ax=ax, edge_labels=edge_labels, font_size=18)
            plt.savefig(f'{parsed.dir}/graph{graph_n}.pdf')
            graph_n += 1
            
            del journalname
            del vertices
            del g
