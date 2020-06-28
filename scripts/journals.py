import numpy as np
import xml.etree.ElementTree as ET
import pickle

arq_o = './drive/My Drive/Colab Notebooks/Mestrado/data/dblp2.xml'

# 0: nao direcionado
# 1: direcionado
# 2: bidirecionado (nao implementado ainda)
opcao_grafo = 0

# Gerador no arquivo teste?
test = False
test_name = ""
if test:
    test_name = "test"
    arq_o = './drive/My Drive/Colab Notebooks/Mestrado/data/dblp_new2.xml'

# dblp = open(arq_o, 'r', encoding="utf-8")
# root = ET.parse(dblp).getroot()
# dblp.close()
root = ET.iterparse(arq_o, events=('start', 'end'))

journals = {}
journals_publications = {}
set_of_authors = set()
if opcao_grafo == 2:
    for event, child in root:
        if event == 'start':
            if child.tag in {"article", "inproceedings"}:
                authors = []
            elif child.tag == "author":
                authors.append(child.text)
                set_of_authors.add(child.text)
            elif child.tag in {"journal", "booktitle"}:
                if child.text in journals:
                    for author in authors:
                        try:
                            index = journals[child.text].index(author)
                            journals_publications[child.text][index] += 1
                        except:
                            journals[child.text].append(author)
                            journals_publications[child.text].append(1)
                            continue
                else:
                    journals[child.text] = authors
                    journals_publications[child.text] = [1]*len(authors)

    with open('./drive/My Drive/Colab Notebooks/Mestrado/data/journals_publications_dict' + test_name + '.pickle', 'wb') as handle:
        pickle.dump(journals_publications, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('./drive/My Drive/Colab Notebooks/Mestrado/data/set_of_authors' + test_name + '.pickle', 'wb') as handle:
        pickle.dump(set_of_authors, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    for event, child in root:
        if event == 'start':
            if child.tag in {"article", "inproceedings"}:
                authors = []
            elif child.tag == "author":
                authors.append(child.text)
            elif child.tag in {"journal", "booktitle"}:
                if child.text in journals:
                    for author in authors:
                        if author not in journals[child.text]:
                            journals[child.text].append(author)
                else:
                    journals[child.text] = authors

    with open('./drive/My Drive/Colab Notebooks/Mestrado/data/journals_dict' + test_name + '.pickle', 'wb') as handle:
        pickle.dump(journals, handle, protocol=pickle.HIGHEST_PROTOCOL)