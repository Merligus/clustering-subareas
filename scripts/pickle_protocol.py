import pickle

test = False
test_name = ""
if test:
    test_name = "test"
    arq_o = 'G:\\Mestrado\\BD\\data\\dblp_new2.xml'

with open('G:\\Mestrado\\BD\\data\\journals_dict' + test_name + '.pickle', 'rb') as handle:
    journals = pickle.load(handle)

with open('G:\\Mestrado\\BD\\data\\journals_dict' + test_name + '.pickle2', 'wb') as handle:
    pickle.dump(journals, handle, protocol=2)