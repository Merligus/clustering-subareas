import numpy as np
import xml.etree.ElementTree as ET
import pickle

arq_o = 'G:\\Mestrado\\BD\\data\\dblp2.xml'

# 0: nao direcionado
# 1: direcionado
# 2: bidirecionado (nao implementado ainda)
opcao_grafo = 2

# Gerador no arquivo teste?
test = False
test_name = ""
if test:
    test_name = "test"
    arq_o = 'G:\\Mestrado\\BD\\data\\dblp_new2.xml'

dblp = open(arq_o, 'r', encoding="utf-8")
root = ET.parse(dblp).getroot()
dblp.close()
journals = {}
journals_publications = {}
set_of_authors = set()

if opcao_grafo == 2:
    for child in root:
        if(child.tag == "article"):
            authors = []
            for attr in child:
                if(attr.tag == "author"):
                    authors.append(attr.text)
                    set_of_authors.add(attr.text)

                if(attr.tag == "journal"):
                    if(attr.text in journals):
                        for author in authors:
                            try:
                                index = journals[attr.text].index(author)
                                journals_publications[attr.text][index] += 1
                            except:
                                journals[attr.text].append(author)
                                journals_publications[attr.text].append(1)
                                continue
                    else:
                        journals[attr.text] = authors
                        journals_publications[attr.text] = [1]*len(authors)

    with open('G:\\Mestrado\\BD\\data\\journals_publications_dict' + test_name + '.pickle', 'wb') as handle:
        pickle.dump(journals_publications, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('G:\\Mestrado\\BD\\data\\set_of_authors' + test_name + '.pickle', 'wb') as handle:
        pickle.dump(set_of_authors, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    for child in root:
        if(child.tag == "article"):
            authors = []
            for attr in child:
                if(attr.tag == "author"):
                    authors.append(attr.text)

                if(attr.tag == "journal"):
                    if(attr.text in journals):
                        for author in authors:
                            if(author not in journals[attr.text]):
                                journals[attr.text].append(author)
                    else:
                        journals[attr.text] = authors

    with open('G:\\Mestrado\\BD\\data\\journals_dict' + test_name + '.pickle', 'wb') as handle:
        pickle.dump(journals, handle, protocol=pickle.HIGHEST_PROTOCOL)