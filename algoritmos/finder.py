import argparse
import igraph as ig
import numpy as np
import pickle
from collections import defaultdict
import nltk
import string

class ClusterFinder:

    def __init__(self, children, n_samples, in_set_conf, labels_sets):
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
        for v in range(self.n_samples):
            self.labels_sets.append({v})
        
        iteration = 0
        for index in range(initial_it, self.children.shape[0]):
            v1, v2 = self.children[index, 0], self.children[index, 1]
            if v1 >= self.n_samples:
                v1 = self.nodes[v1-self.n_samples]
            if v2 >= self.n_samples:
                v2 = self.nodes[v2-self.n_samples]
            self.labels_sets[v1] = self.labels_sets[v1].union(self.labels_sets[v2])
            found = True
            for v in in_set_conf:
                if v not in self.labels_sets[v1]:
                    found = False
                    break
            
            self.nodes.append(v1)
            self.labels_sets[v2] = {-1}

            if found:
                print("Achou")
                iteration = index
                self.v = v1
                break
            
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

parsed = parser.parse_args()
if parsed.in_name != '-':
    parsed.in_name = '_' + parsed.in_name
else:
    parsed.in_name = ''
print(25*'*', 'ARGS', 25*'*')
print(f'Mode: {parsed.mode}')
print(f'In name: {parsed.in_name}')
print(f'Directory: {parsed.dir}')

filename = f"{parsed.dir}/graph_nao_direcionado" + parsed.mode + parsed.in_name + '.npy'
with open(filename, "rb") as f:
    adj_mat = np.load(f)

filename = f"{parsed.dir}/children_agglomerative_" + parsed.mode + '.npy'
with open(filename, "rb") as f:
    children = np.load(f)

with open(f'{parsed.dir}/index_to_journalname.pickle', 'rb') as handle:
    index_to_journalname = pickle.load(handle)

with open(f'{parsed.dir}/index_to_journal_complete_name.pickle', 'rb') as handle:
    index_to_journal_complete_name = pickle.load(handle)

with open(f'{parsed.dir}/journals_dict{parsed.in_name}.pickle', 'rb') as handle:
    journals = pickle.load(handle)

# index_to_journal_complete_name = {}
# for v in index_to_journalname:
#     journal = index_to_journalname[v]
#     if journal not in journals:
#         print(journal)
#         continue
#     if len(journals[journal]['journal_name']) > 0:
#         index_to_journal_complete_name[v] = journal + ': ' + journals[journal]['journal_name']
#     elif 'journal_name_rough' in journals[journal]:
#         index_to_journal_complete_name[v] = journal + ': ' + journals[journal]['journal_name_rough']

# with open(f'{parsed.dir}/index_to_journal_complete_name.pickle', 'wb') as handle:
#     pickle.dump(index_to_journal_complete_name, handle, protocol=2)

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
    print(f'Identificados: {s}')

    n_samples = adj_mat.shape[0]
    
    print("Procurando...")
    labels_sets = []
    cf = ClusterFinder(children, n_samples, in_set_conf, labels_sets)
    c, iteration = cf.find_cluster(0)
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
        
        # opcao de sair da atual
        if entrada[0] == '0':
            break
        # opcao de mostrar a proxima juncao do cluster
        elif entrada[0] == '1':
            c, iteration = cf.find_cluster(iteration+1)
            cluster = cf.labels_sets[c]
            print(f"Cluster de tamanho {len(cluster)} na iteração {iteration}")
        # opcao de mostrar o cluster
        elif entrada[0] == '2':
            for i, vi in enumerate(cluster):
                print(f"{i}: {index_to_journal_complete_name[vi]}")
        # mostra a frequencia das palavras no cluster
        elif entrada[0] == '3':
            sentences = []
            for vi in cluster:
                sentences.append(index_to_journal_complete_name[vi].lower())
            cf.show_top(sentences, n=10)
        # plota o cluster em forma de grafo
        elif entrada[0] == '4':
            g = ig.Graph()
            g.add_vertices(len(cluster))
            edges = []
            weights = []

            vertices = []
            for vi in cluster:
                vertices.append(vi)

            journalname = []
            v1 = 0
            while v1 < len(cluster):
                v2 = v1 + 1
                while v2 < len(cluster):
                    if adj_mat[vertices[v1], vertices[v2]] > 0 and adj_mat[vertices[v1], vertices[v2]] < np.inf:
                        edges.append((v1, v2))
                        weights.append(adj_mat[vertices[v1], vertices[v2]])
                    v2 += 1
                journalname.append(index_to_journalname[vertices[v1]])
                v1 += 1
            g.add_edges(edges)
            g.es["weights"] = weights
            g.vs["label"] = journalname

            layout = g.layout("kk")
            ig.plot(g, layout=layout, target=f'graph.pdf')
            
            del journalname
            del vertices
            del g
            del weights
            del edges
