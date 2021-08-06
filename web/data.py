import numpy as np
import pickle
from algoritmos.finder import ClusterFinder

class Data:
    def __init__(self, in_name, mode, dir):
        self.children = {}
        for function in ["agglomerative", "multilevel"]:
            filename = f"{dir}/children_{function}_" + mode + in_name
            with open(filename + '.pickle', 'rb') as f:
                self.children[function] = pickle.load(f)

        filename = f"{dir}/graph_nao_direcionado" + mode + in_name + '.npy'
        with open(filename, "rb") as f:
            self.adj_mat = np.load(f)
        self.distance = np.nanmin(self.adj_mat) + np.nanmax(self.adj_mat) - self.adj_mat

        with open(f'{dir}/index_to_journalname_{mode}{in_name}.pickle', 'rb') as handle:
            self.index_to_journalname = pickle.load(handle)

        with open(f'{dir}/journals_dict{in_name}.pickle', 'rb') as handle:
            self.journals = pickle.load(handle)

        self.index_to_journal_complete_name = {}
        for v in self.index_to_journalname:
            journal = self.index_to_journalname[v]
            if journal not in self.journals:
                print(journal)
                continue
            suff = ""
            if len(self.journals[journal]['journal_name']) > 0:
                suff += " " + self.journals[journal]['journal_name']
            if 'journal_name_rough' in self.journals[journal]:
                suff += " -- " + self.journals[journal]['journal_name_rough']
            self.index_to_journal_complete_name[v] = journal + ':' + suff

        with open(f'{dir}/index_to_journal_complete_name{in_name}.pickle', 'wb') as handle:
            pickle.dump(self.index_to_journal_complete_name, handle, protocol=2)

        with open(f'{dir}/index_to_journal_complete_name{in_name}.pickle', 'rb') as handle:
            self.index_to_journal_complete_name = pickle.load(handle)

        with open(f'{dir}/nauthors{in_name}.pickle', 'rb') as handle:
            self.nauthors = pickle.load(handle)

        self.journal_names_list = []
        with open(f'{dir}/journal_names.txt') as fr:
            for line in fr:
                final_ind = line.find(':')
                if line[:final_ind] in self.journals:
                    self.journal_names_list.append((line[:final_ind], line[final_ind+1:-1]))

class Params:
    def __init__(self, in_name, mode, dir, function, n_samples, iteration=-1, 
                 children=[], in_set_conf=[0]):
        
        self.in_name = in_name
        if self.in_name != '-':
            self.in_name = '_' + self.in_name
        else:
            self.in_name = ''
        self.mode = mode
        self.dir = dir
        self.function = function
        
        self.cf = ClusterFinder(children, n_samples, set(in_set_conf), [])
        self.iteration = iteration
        self.old_cluster = set(in_set_conf)
        
        if self.iteration >= 0:
            c, aux_iteration = self.cf.find_cluster(0)
            self.cluster = self.cf.labels_sets[c]
            while aux_iteration != self.iteration:
                self.old_cluster = self.cluster
                c, aux_iteration = self.cf.find_cluster(aux_iteration+1)
                if self.iteration < len(self.cf.children):
                    self.cluster = self.cf.labels_sets[c]
                elif aux_iteration == len(self.cf.children):
                    print(f"Nenhum cluster achado")

        print(25*'*', 'ARGS', 25*'*')
        print(f'Mode: {self.mode}')
        print(f'In name: {self.in_name}')
        print(f'Directory: {self.dir}')
        print(f'Function: {self.function}')
        