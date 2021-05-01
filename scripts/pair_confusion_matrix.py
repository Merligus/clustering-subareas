from sklearn import metrics

names = ['fastgreedy', 'infomap', 'leading_eigenvector', 'multilevel', 'walktrap', 'label_propagation', 'community_leiden']
posts = ['mean', 'min', 'union']

for name in names:
    for post in posts:
        arq_path_no_log = f'G:\Mestrado\BD\data\saidas\saida_{name}_0_{post}.out'
        arq_path_log = f'G:\Mestrado\BD\data\saidas\saida_{name}_1_{post}.out'

        with open(arq_path_no_log, 'r') as f:
            next = False
            for line in f:
                if 'labels:' in line:
                    next = True
                elif next:
                    labels_no_log = line[1:-2].split(',')
                    labels_no_log = [int(li) for li in labels_no_log]
                    break
        with open(arq_path_log, 'r') as f:
            next = False
            for line in f:
                if 'labels:' in line:
                    next = True
                elif next:
                    labels_log = line[1:-2].split(',')
                    labels_log = [int(li) for li in labels_log]
                    break

        print(f'{name} {post}\n\tPair confusion matrix: \n{metrics.cluster.pair_confusion_matrix(labels_log, labels_no_log)}')