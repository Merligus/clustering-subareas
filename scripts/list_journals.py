import pickle

comb = []
for cut in [0, 0.2]:
    for only_journals in [0, 1]:
        for year in [0, 2010]:
            if only_journals and cut > 0:
                continue
            in_name = ""
            if year > 0:
                in_name += '_' + str(year)
            if only_journals:
                in_name += '_only_journals'
            if cut > 0:
                in_name += f'_cut{cut:.3}'
            comb.append(in_name)

for in_name in comb:
    print('ABRINDO ARQUIVO')
    with open(f'../data/journals_dict{in_name}.pickle', 'rb') as handle:
        journals = pickle.load(handle)
    print('ARQUIVO ABERTO')
    for function, rec in [('multilevel', 4)]: #, ('agglomerative', 0)]:
        for mode in ['union', 'max', 'mean']:
            fin = open(f'../data/{function}_{mode}_rec{rec}{in_name}.txt')
            fout = open(f'../data/formatted_output/{function}_{mode}_rec{rec}{in_name}.txt', 'w')
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
