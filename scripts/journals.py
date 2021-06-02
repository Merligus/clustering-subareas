import numpy as np
import xml.etree.ElementTree as ET
import pickle
import nltk
from collections import defaultdict
import sys

arq_o = '../data/dblp2.xml'

# 0: nao direcionado
# 1: direcionado
# 2: bidirecionado (nao implementado ainda)
opcao_grafo = 0

# Gerador no arquivo teste?
if (len(sys.argv) < 3):
    print('Falta parametros')
    exit()
elif (len(sys.argv) > 3):
    print('Muitos parametros')
    exit()
else:
    only_journals = bool(int(sys.argv[1]))
    cut = int(sys.argv[2])
test = False
year = 0
test_name = ""
if test:
    test_name = "test"
    arq_o = '../data/dblp_new2.xml'
if year > 0:
    test_name += '_' + str(year)
if only_journals:
    test_name += '_only_journals'
if cut > 0:
    test_name += '_cut' + str(cut)

# dblp = open(arq_o, 'r', encoding="utf-8")
# root = ET.parse(dblp).getroot()
# dblp.close()
root = ET.iterparse(arq_o, events=('start', 'end'))

journals = {}
quantity_articles = defaultdict(int)
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
    publtype_not_accepted = {'informal', 'software', 'informal withdrawn', 'survey', 'withdrawn', 'data', 'edited'}
    publtype = {}
    authors = []
    tag = ''
    journal_name = '-'
    save = 0x00
    for event, child in root:
        if event == 'start':
            # <article mdate="2020-03-12" key="tr/meltdown/m18" publtype="informal">
            if child.tag in {"article", "inproceedings", "proceedings"}:
                tag = child.tag
                authors = []
                journal_name = '-'
                if 'publtype' not in child.attrib:
                    save = 0x04 # can save
                elif child.attrib['publtype'] not in publtype_not_accepted:
                    save = 0x04 # can save
                    publtype[child.attrib['publtype']] = True
                else:
                    save = 0x00 # cannot save
            elif child.tag == "title" and child.text is not None and tag == 'proceedings':
                journal_name = child.text # add journal name
            elif child.tag == "author":
                authors.append(child.text) # add author
            elif child.tag == "year" and child.text:
                if int(child.text) >= year:
                    save = save | 0x01 # can save
            # <url>db/conf/cmcs/cmcs2001.html#Hughes01</url>
            elif child.tag == "url" and child.text is not None and not (save & 0x02): # save == bxxx1x = cross ja achou journal
                url = child.text
                find_c = child.text.find('conf')
                shift_c = 4 # len('conf')
                find_j = child.text.find('journals')
                shift_j = 8 # len('journals')
                if find_c > -1 or find_j > -1:
                    if find_c > -1 and not only_journals:
                        save = save | 0x02 # can save
                        start = find_c + shift_c + 1
                        end = start + url[start:].find('/')
                        journal = url[start : end]
                    else:
                        save = save | 0x02 # can save
                        start = find_j + shift_j + 1
                        end = start + url[start:].find('/')
                        journal = url[start : end]                    
            # <crossref>conf/cmcs/2001</crossref>
            elif child.tag == "crossref" and child.text is not None:
                cross = child.text
                find_c = child.text.find('conf')
                shift_c = 4 # len('conf')
                find_j = child.text.find('journals')
                shift_j = 8 # len('journals')
                if find_c > -1 or find_j > -1:
                    if find_c > -1 and not only_journals:
                        save = save | 0x02 # can save
                        start = find_c + shift_c + 1
                        end = start + cross[start:].find('/')
                        journal = cross[start : end]
                    else:
                        save = save | 0x02 # can save
                        start = find_j + shift_j + 1
                        end = start + cross[start:].find('/')
                        journal = cross[start : end]
        elif event == 'end' and save == 0x07 and child.tag in {'article', 'inproceedings', 'proceedings'}:
            save = 0x00
            if journal not in journals:
                journals[journal] = {}
                journals[journal]['journal_name'] = []
            if child.tag in {"article", "inproceedings"}:
                quantity_articles[journal] += 1
                for author in authors:
                    journals[journal][author] = True
            elif child.tag in {"proceedings"}:
                if journal_name[0] != '-':
                    journals[journal]['journal_name'].append(journal_name)
        if event == 'end':
            child.clear()

    remove_list = []
    for journal in journals:
        if quantity_articles[journal] < cut:
            remove_list.append(journal)
            continue
        if len(journals[journal]['journal_name']) > 0:
            # Nome completo com metadados
            ocurrences = defaultdict(int)
            for journal_name in journals[journal]['journal_name']:
                tokens = nltk.word_tokenize(journal_name) 
                for t in tokens:
                    ocurrences[t] += 1
            relevant_tokens = []
            for t in ocurrences:
                if ocurrences[t] >= len(journals[journal]['journal_name'])/2:
                    relevant_tokens.append(t)
            # pega o nome que aparece mais tokens
            name = ''
            max_c = 0
            for journal_name in journals[journal]['journal_name']:
                count = 0
                for token in relevant_tokens:
                    if token in journal_name:
                        count += 1
                if count > max_c:
                    name = journal_name
                    max_c = count
            # monta o nome
            name_tokenized = nltk.word_tokenize(name)
            name = ''
            relevant_tokens = set(relevant_tokens)
            for token in name_tokenized:
                if token in relevant_tokens:
                    name += token + ' ' 
            
            journals[journal]['journal_name'] = name
        else:
            journals[journal]['journal_name'] = ''

    for journal in remove_list:
        journals.pop(journal, None)
        
    print(f'publtypes = {publtype.keys()}')

    with open('../data/journal_names.txt') as fr:
        for line in fr:
            final_ind = line.find(':')
            if line[:final_ind] in journals:
                journals[line[:final_ind]]['journal_name_rough'] = line[final_ind+1:-1]

    with open('../data/journals_dict' + test_name + '.pickle', 'wb') as handle:
        pickle.dump(journals, handle, protocol=2)
