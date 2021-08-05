import numpy as np
import pickle
from algoritmos.finder import ClusterFinder

class Params():
    def __init__(self, in_name, mode, dir, function):
        
        self.in_name = in_name
        if self.in_name != '-':
            self.in_name = '_' + self.in_name
        else:
            self.in_name = ''
        self.mode = mode
        self.dir = dir
        self.function = function

        self.distance = np.zeros((2,2))
        self.adj_mat = np.zeros((2,2))
        self.children = [2, 2]
        self.index_to_journalname = [2, 2]
        self.journals = {'a': [1, 2], 'b':[3, 4]}
        self.index_to_journal_complete_name = {'a': [1, 2], 'b':[3, 4]}
        self.nauthors = [2, 2]
        self.iteration = 0
        self.old_cluster = {'a': [1, 2], 'b':[3, 4]}
        self.cluster = {'a': [1, 2], 'b':[3, 4]}
        self.cf = ClusterFinder(self.children, len(self.distance), set([0]), [])

        print(25*'*', 'ARGS', 25*'*')
        print(f'Mode: {self.mode}')
        print(f'In name: {self.in_name}')
        print(f'Directory: {self.dir}')
        print(f'Function: {self.function}')
    
    def serialize(self):
        return
    
    def deserialize(self):
        return

    def load_files(self):
        filename = f"{self.dir}/graph_nao_direcionado" + self.mode + self.in_name + '.npy'
        with open(filename, "rb") as f:
            self.adj_mat = np.load(f)
        self.distance = np.nanmin(self.adj_mat) + np.nanmax(self.adj_mat) - self.adj_mat

        filename = f"{self.dir}/children_{self.function}_" + self.mode + self.in_name
        if self.function == 'agglomerative': # load em python
            with open(filename + '.npy', "rb") as f:
                self.children = np.load(f)
        elif self.function == 'multilevel':
            with open(filename + '.pickle', 'rb') as f:
                self.children = pickle.load(f)
        else:
            raise KeyError(f'Função não disponível.')

        with open(f'{self.dir}/index_to_journalname_{self.mode}{self.in_name}.pickle', 'rb') as handle:
            self.index_to_journalname = pickle.load(handle)

        with open(f'{self.dir}/journals_dict{self.in_name}.pickle', 'rb') as handle:
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

        with open(f'{self.dir}/index_to_journal_complete_name{self.in_name}.pickle', 'wb') as handle:
            pickle.dump(self.index_to_journal_complete_name, handle, protocol=2)

        with open(f'{self.dir}/index_to_journal_complete_name{self.in_name}.pickle', 'rb') as handle:
            self.index_to_journal_complete_name = pickle.load(handle)

        with open(f'{self.dir}/nauthors{self.in_name}.pickle', 'rb') as handle:
            self.nauthors = pickle.load(handle)
        
        self.iteration = 0
        self.old_cluster = set()
        self.cluster = set()
        self.cf = ClusterFinder(self.children, len(self.distance), set([0]), [])
        