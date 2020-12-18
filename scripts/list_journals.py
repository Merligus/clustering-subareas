import pickle

with open('G:\\Mestrado\\BD\\data\\journals_dict.pickle', 'rb') as handle:
    journals = pickle.load(handle)

f = open('G:\\Mestrado\\BD\\data\\journal_listed.txt', 'w')
for journal in journals:
    try:
        if type(journal) != 'NoneType':
            f.write(journal + '\n')
    except: 
        a = 1 
f.close()