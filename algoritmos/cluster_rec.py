from igraph import *

class ClusterRec:

    def __init__(self, function="multilevel", threshold=0, times=3):
        self.function = function
        self.threshold = threshold
        self.times = times

    def fit(self, graph):
        V = len(graph.vs.indegree())
        graph['names'] = [v for v in range(V)]
        graphs = [('', graph)]
        self.VC = []
        for _ in range(self.times):
            new_graphs = []
            for ind, g in graphs:
                # Vertex Cluster
                if self.function == 'fastgreedy':
                    vd = g.community_fastgreedy(weights=g.es['commonauthors'])
                    vc = vd.as_clustering()
                elif self.function == 'infomap':
                    vc = g.community_infomap(edge_weights=g.es['commonauthors'])
                elif self.function == 'leading_eigenvector':
                    vc = g.community_leading_eigenvector(weights=g.es['commonauthors'])
                elif self.function == 'multilevel':
                    vc = g.community_multilevel(weights=g.es['commonauthors'])
                elif self.function == 'walktrap':
                    vd = g.community_walktrap(weights=g.es['commonauthors'])
                    vc = vd.as_clustering()
                elif self.function == 'optimal_modularity':
                    vc = g.community_optimal_modularity(weights=g.es['commonauthors'])
                elif self.function == 'edge_betweenness':
                    vd = g.community_edge_betweenness(weights=g.es['commonauthors'], directed=False)
                    vc = vd.as_clustering()
                elif self.function == 'spinglass':
                    vc = g.community_spinglass(weights=g.es['commonauthors'])
                elif self.function == 'label_propagation':
                    initial_temp = {}
                    for vid in range(len(g.vs.indegree())):
                        if g.vs[vid]['fixed']:
                            if g.vs[vid]['initial'] not in initial_temp:
                                initial_temp[g.vs[vid]['initial']] = len(initial_temp)
                            g.vs[vid]['initial'] = initial_temp[g.vs[vid]['initial']]
                    vc = g.community_label_propagation(weights=g.es["commonauthors"], initial=g.vs["initial"], fixed=g.vs["fixed"])
                elif self.function == 'community_leiden':
                    initial_temp = {}
                    for vid in range(len(g.vs.indegree())):
                        if g.vs[vid]['fixed']:
                            if g.vs[vid]['initial'] not in initial_temp:
                                initial_temp[g.vs[vid]['initial']] = len(initial_temp)
                            g.vs[vid]['initial'] = initial_temp[g.vs[vid]['initial']]
                    vc = g.community_leiden(weights=g.es["commonauthors"], initial_membership=g.vs["initial"])
                else:
                    print('FUNCTION NOT RECOGNIZED')
                    return self
                # For each community
                for e, comm in enumerate(vc):
                    # Too big, execute it again
                    if len(comm) > self.threshold:
                        # Create the subgraph of the community
                        sub_g = g.subgraph(comm, implementation='create_from_scratch')
                        # Saving their real names
                        sub_g['names'] = [g['names'][v] for v in comm]
                        # Insert the subgraph in the queue of execution
                        new_graphs.append((ind + str(e + 1) + '.', sub_g))
                    else:
                        # No need to execute it again, just save their real names
                        self.VC.append((ind + str(e + 1) + '.', [g['names'][v] for v in comm]))
            del graphs
            graphs = new_graphs
        for ind, g in graphs:
            self.VC.append((ind, g['names']))
        return self
