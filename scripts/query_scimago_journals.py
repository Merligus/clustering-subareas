import requests
import csv

# SCRIPT PARA PEGAR OS NOMES DOS VEICULOS PELA API DO DBLP

for area in ['APP', 'CG', 'CV', 'HCI', 'HW', 'IS', 'NET', 'SP', 'SW', 'TM']:
    filename = f"scimago_{area}"
    print(filename)
    with open(f'../data/{filename}.csv') as csvfile:
        fr = csv.reader(csvfile, delimiter=';')
        fw = open(f'../data/journal_names_{filename}.txt', 'w', encoding="utf-8")
        journals = {}
        
        for row in fr:
            journal_name_query = row[2]
            params = {'q' : journal_name_query, 'format' : 'json', 'h' : '1000', 'c' : '0'}
            jason_object = requests.get('https://dblp.org/search/venue/api', params=params).json()
            
            if 'hit' not in jason_object['result']['hits']:
                print(f'\tnao encontrado {journal_name_query}')
                continue
            
            # Nome completo sem metadados
            found = False
            for hit in jason_object['result']['hits']['hit']:
                journal_name = hit['info']['venue']
                if journal_name == journal_name_query:
                    url = hit['info']['url']
                    fim_i = url.rfind('/')
                    ini_i = url[:fim_i].rfind('/')
                    sigla = url[ini_i+1:fim_i]

                    find_c = url.find('/conf/')
                    find_j = url.find('/journals/')
                    suf = '_'
                    suf += 'c' if find_c > 0 else ''
                    suf += 'j' if find_j > 0 else ''
                    sigla += suf

                    if find_c > 0 or find_j > 0:
                        if sigla not in journals:
                            journals[sigla] = True
                            found = True
                            fw.write(f'{sigla}:{journal_name}\n')
            if found:
                print(f'\tENCONTRADO {journal_name_query}')
            else:
                print(f'\tnao encontrado {journal_name_query}')
        fw.close()