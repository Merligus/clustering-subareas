#!/usr/bin/env python3
import pickle

print('ABRINDO ARQUIVO')
with open('../data/journals_dict.pickle', 'rb') as handle:
    journals = pickle.load(handle)
print('ARQUIVO ABERTO')

for TIMES in [300, 500]:
    fin = open(f'../data/agglomerative_link_average_n_clusters_{TIMES}.txt')
    fout = open(f'../data/agglomerative_link_average_n_clusters_{TIMES}_.txt', 'w')
    for line in fin:
        if line[0] == '\t':
            id = line.find(':')
            journal = line[id+1:-1]
            try:
                fout.write('\t' + line[1:-1] + ':' + journals[journal]['journal_name'] + '\n')
            except:
                try:
                    fout.write('\t' + line[1:-1] + ':' + journals[journal]['journal_name_rough'] + '\n')
                except:
                    fout.write('\t' + line[1:-1] + ':' + '\n')
                print(line[1:-1])
        else:
            fout.write(line)
    fin.close()
    fout.close()

# f = open('../data/journal_listed.txt', 'w')
# print('ESCREVENDO')
# for journal in journals:
#     try:
#         if type(journal) != 'NoneType':
#             f.write(journal + ':' + journals[journal]['journal_name']  + '\n')
#     except:
#         a = 1
# f.close()
