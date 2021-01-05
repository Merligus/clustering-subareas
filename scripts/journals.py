import numpy as np
import xml.etree.ElementTree as ET
import pickle

arq_o = '../data/dblp2.xml'

# 0: nao direcionado
# 1: direcionado
# 2: bidirecionado (nao implementado ainda)
opcao_grafo = 0

# Gerador no arquivo teste?
test = False
year = 0
test_name = ""
if test:
    test_name = "test"
    arq_o = '../data/dblp_new2.xml'
if year > 0:
    test_name += '_' + str(year)


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
                            if len(author) > 0:
                                index = journals[child.text].index(author)
                                journals_publications[child.text][index] += 1
                        except:
                            if len(author) > 0:
                                journals[child.text].append(author)
                                journals_publications[child.text].append(1)
                            continue
                else:
                    journals[child.text] = authors
                    journals_publications[child.text] = [1]*len(authors)


    with open('../data/journals_publications_dict' + test_name + '.pickle', 'wb') as handle:
        pickle.dump(journals_publications, handle, protocol=2)
    
    with open('../data/set_of_authors' + test_name + '.pickle', 'wb') as handle:
        pickle.dump(set_of_authors, handle, protocol=2)
else:
    save = 0x00
    publtype_not_accepted = ['informal', 'software', 'informal withdrawn', 'survey', 'withdrawn', 'data', 'edited']
    publtype = {}
    for event, child in root:
        if event == 'start':
            # <article mdate="2020-03-12" key="tr/meltdown/m18" publtype="informal">
            if child.tag in {"article", "inproceedings"}:
                authors = []
                if 'publtype' not in child.attrib:
                    save = 0x04 # can save
                elif child.attrib['publtype'] not in publtype_not_accepted:
                    save = 0x04 # can save
                    publtype[child.attrib['publtype']] = True
                else:
                    save = 0x00 # cannot save
            elif child.tag == "author":
                authors.append(child.text) # add author
            # elif child.tag in {"journal", "booktitle"}:
            #     journal = child.text # add journal name
            elif child.tag == "year" and child.text:
                if int(child.text) >= year:
                    save = save | 0x01 # can save
            # <url>db/conf/cmcs/cmcs2001.html#Hughes01</url>
            elif child.tag == "url" and child.text is not None and not (save & 0x02): # not (save & 0x02) serve pra verificar se cross ja achou journal
                url = child.text
                find_c = child.text.find('conf')
                shift_c = len('conf')
                find_j = child.text.find('journals')
                shift_j = len('journals')
                if find_c > -1 or find_j > -1:
                    save = save | 0x02 # can save
                    if find_c > -1:
                        start = find_c + shift_c + 1
                        end = start + url[start:].find('/')
                        journal = url[start : end]
                    else:
                        start = find_j + shift_j + 1
                        end = start + url[start:].find('/')
                        journal = url[start : end]                    
            # <crossref>conf/cmcs/2001</crossref>
            elif child.tag == "crossref" and child.text is not None:
                cross = child.text
                find_c = child.text.find('conf')
                shift_c = len('conf')
                find_j = child.text.find('journals')
                shift_j = len('journals')
                if find_c > -1 or find_j > -1:
                    save = save | 0x02 # can save
                    if find_c > -1:
                        start = find_c + shift_c + 1
                        end = start + cross[start:].find('/')
                        journal = cross[start : end]
                    else:
                        start = find_j + shift_j + 1
                        end = start + cross[start:].find('/')
                        journal = cross[start : end]
        elif event == 'end' and child.tag in {'article', 'inproceedings'} and save == 0x07:
            save = 0x00
            if journal in journals:
                for author in authors:
                    journals[journal][author] = True
            else:
                journals[journal] = {}
                for author in authors:
                    journals[journal][author] = True

    print(f'publtypes = {publtype.keys()}')
    with open('../data/journals_dict' + test_name + '.pickle', 'wb') as handle:
        pickle.dump(journals, handle, protocol=2)

