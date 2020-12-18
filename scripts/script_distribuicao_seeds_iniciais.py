names = ['_log_min.out', '_log_mean.out', '_log_union.out']
for name in names:
    arq_path = f'G:\Mestrado\BD\data\out\saida{name}'
    with open(arq_path, 'r') as f:
        s1 = ' da comunidade '
        shift1 = len(s1)
        s2 = ' foi para a comunidade '
        shift2 = len(s2)
        freq = []
        freq_d = {}
        for i in range(17):
            freq.append({})
        for line in f:
            if ' da comunidade' in line:
                i = line.find(s1)
                j = line.find(s2)
                ini = int(line[i+shift1 : i+shift1+2])
                fin = int(line[j+shift2 : len(line)])
                # print(line.strip(), ini, fin)
                if fin in freq[ini]:
                    freq[ini][fin] += 1
                else:
                    freq[ini][fin] = 1
            elif line[:4] == 'Fim ':
                freq_d[line[4:-1]] = freq
                freq = []
                for i in range(17):
                    freq.append({})
        print(freq_d)

    with open(arq_path, 'a') as f:
        f.write('\n<seed>\n')
        f.write('\t<quantidade> na <comunidade>\n')
        for alg in freq_d:
            freq = freq_d[alg]
            f.write(f'{alg}\n')
            for seed, d in enumerate(freq):
                f.write(f'\t{seed}\n')
                for comunidade in d:
                    f.write(f'\t\t{d[comunidade]} na {comunidade}\n')