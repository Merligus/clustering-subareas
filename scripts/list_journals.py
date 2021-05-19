import pickle

print('ABRINDO ARQUIVO')
with open('../data/journals_dict.pickle', 'rb') as handle:
    journals = pickle.load(handle)
print('ARQUIVO ABERTO')

# quando eh o MDS
comb = []
for w_o in ['normal', '1or0', 'd-1', 'd-2']:
    for n_comps in [2, 3, 4, 5, 6, 7]:
        for n_clus in [20, 40]:
            comb.append((w_o, n_comps, n_clus))

for rec in [2, 3]:
    fin = open(f'../data/multilevel_union_rec{rec}.txt')
    fout = open(f'../data/formatted_output/multilevel_union_rec{rec}.txt', 'w')
    for line in fin:
        if line[0] == '\t':
            id = line.find(':')
            journal = line[id+1:-1]
            if len(journals[journal]['journal_name']) > 0 :
                fout.write('\t' + line[1:-1] + ':' + journals[journal]['journal_name'] + '\n')
            elif len(journals[journal]['journal_name_rough']) > 0:
                fout.write('\t' + line[1:-1] + ':' + journals[journal]['journal_name_rough'] + '\n')
            else:
                fout.write('\t' + line[1:-1] + ':' + journal.upper() + '\n')
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
