names = ['Fastgreedy', 'Leading_eigenvector', 'Walktrap', 'Multilevel', 'Infomap']
# posts = ['_min_log.txt', '_mean_log.txt']
posts = ['_union_log.txt']
for name in names:
    for post in posts:
        arq_path = f'G:\Mestrado\BD\data\log\{name}{post}'        
        with open(arq_path, 'r') as f:
            lista_index = []
            tamanho = []
            count = 0
            for index, line in enumerate(f):
                if line[0] != '\t':
                    tamanho.append(count)
                    count = 0
                else:
                    count += 1
                    # lista_index.append(index)
                    # # print(index, line.strip())
                    # if index != 0:
                    #     tamanho.append(index - lista_index[-2] - 1)
            tamanho.append(count)
            print(tamanho[1:])

        with open(arq_path, 'a') as f:
            f.write('\n<comunidade> <tamanho\n')
            for i, t in enumerate(tamanho[1:]):
                f.write(f'{i} com {t} revistas\n')