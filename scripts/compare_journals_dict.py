import numpy as np
import pickle


name1 = 'journals_dict.pickle'
with open('../data/journals_dict.pickle', 'rb') as handle:
    journals1 = pickle.load(handle)

name2 = 'journals_dict_no_proceedings.pickle'
with open('../data/journals_dict_no_proceedings.pickle', 'rb') as handle:
    journals2 = pickle.load(handle)

feito = {}
remove = []
for journal1 in journals1:
    if 'journal_name' in journals1[journal1]:
        journals1[journal1].pop('journal_name', None)
    if 'journal_name_rough' in journals1[journal1]:
        journals1[journal1].pop('journal_name_rough', None)

    if journal1 not in journals2:
        print(f'{journal1} not in {name2}:tamanho = {len(journals1[journal1])}')
        remove.append(journal1)
        continue
    else:
        feito[journal1] = True

    if len(journals1[journal1]) != len(journals2[journal1]):
        print(f'{journal1} {len(journals1[journal1])} != {len(journals2[journal1])}')

for journal2 in journals2:
    if journal2 not in feito:
        print(f'{journal2} not in {name1}')

for ri in remove:
    journals1.pop(ri, None)
    print(f'{ri} removido')

print(journals1 == journals2)
