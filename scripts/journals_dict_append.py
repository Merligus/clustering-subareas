import pickle

## SCRIPT PARA JUNTAR journals[<journal>]['journal_name_rough'] a journals[<journal>]['journal_name']

print('ABRINDO ARQUIVO')
with open('G:\\Mestrado\\BD\\data\\journals_dict.pickle', 'rb') as handle:
    journals = pickle.load(handle)
print('ARQUIVO ABERTO')

try:
    with open('G:\\Mestrado\\BD\\data\\journal_names.txt') as fr:
        for line in fr:
            final_ind = line.find(':')
            journals[line[:final_ind]]['journal_name_rough'] = line[final_ind+1:-1]
except:
    print('ERRO')
    exit()

with open('G:\\Mestrado\\BD\\data\\journals_dict.pickle', 'wb') as handle:
    pickle.dump(journals, handle, protocol=2)