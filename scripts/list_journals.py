import pickle

# quando eh o MDS
comb = []
for w_o in ['normal', '1or0', 'd-1', 'd-2']:
    for n_comps in [2, 3, 4, 5, 6, 7]:
        for n_clus in [20, 40]:
            comb.append((w_o, n_comps, n_clus))

for in_name in ['']:
    print('ABRINDO ARQUIVO')
    with open(f'../data/journals_dict_{in_name}.pickle', 'rb') as handle:
        journals = pickle.load(handle)
    print('ARQUIVO ABERTO')
    for function in ['multilevel']:
        for rec in [3]:
            fin = open(f'../data/{function}_union_rec{rec}_{in_name}.txt')
            fout = open(f'../data/formatted_output/{function}_union_rec{rec}_{in_name}.txt', 'w')
            for line in fin:
                if line[0] == '\t':
                    id = line.find(':')
                    journal = line[id+1:-1]
                    if journal not in journals:
                        print(journal, line)
                        continue
                    if len(journals[journal]['journal_name']) > 0:
                        fout.write('\t' + line[1:-1] + ':' + journals[journal]['journal_name'] + '\n')
                    elif 'journal_name_rough' in journals[journal]:
                        fout.write('\t' + line[1:-1] + ':' + journals[journal]['journal_name_rough'] + '\n')
                    # elif 'journal_name' in journals[journal]:
                    #     fout.write('\t' + line[1:-1] + ':' + journals[journal]['journal_name'] + '\n')
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
