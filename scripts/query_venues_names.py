import requests

## SCRIPT PARA PEGAR OS NOMES DOS VEÍCULOS PELA API DO DBLP

journals = {}
fw = open('G:\\Mestrado\\BD\\data\\journal_names.txt', 'w', encoding="utf-8")
with open('G:\\Mestrado\\BD\\data\\journal_listed.txt') as fr:
    for line in fr:
        final_ind = line.find(':')
        journals[line[:final_ind]] = False

query = ''
for i, journal in enumerate(journals.keys()):
    if (i + 1) % 50 == 0:
        params = {'q' : query, 'format' : 'json', 'h' : '1000', 'c' : '0'}
        try:
            jason_object = requests.get('https://dblp.org/search/venue/api', params=params).json()
        except:
            print(f'nao encontrado {query}')
        # Nome completo sem metadados
        for hit in jason_object['result']['hits']['hit']:
            journal_name = hit['info']['venue']
            url = hit['info']['url']
            fim_i = url.rfind('/')
            ini_i = url[:fim_i].rfind('/')
            sigla = url[ini_i+1:fim_i]
            if sigla in journals:
                if not journals[sigla]:
                    journals[sigla] = True
                    fw.write(f'{sigla}:{journal_name}\n')
        query = ''
    else:
        if '-' not in journal:
            query += journal + '$|'

params = {'q' : query, 'format' : 'json', 'h' : '1000', 'c' : '0'}
jason_object = requests.get('https://dblp.org/search/venue/api', params=params).json()
# Nome completo sem metadados
for hit in jason_object['result']['hits']['hit']:
    journal_name = hit['info']['venue']
    url = hit['info']['url']
    fim_i = url.rfind('/')
    ini_i = url[:fim_i].rfind('/')
    sigla = url[ini_i+1:fim_i]
    if sigla in journals:
        if not journals[sigla]:
            journals[sigla] = True
            fw.write(f'{sigla}:{journal_name}\n')

for journal in journals:
    if not journals[journal]:
        print(f'{journal} nao encontrado')
        params = {'q' : journal + '$', 'format' : 'json', 'h' : '1000', 'c' : '0'}
        jason_object = requests.get('https://dblp.org/search/venue/api', params=params).json()
        if 'hit' in jason_object['result']['hits']:
            for hit in jason_object['result']['hits']['hit']:
                journal_name = hit['info']['venue']
                url = hit['info']['url']
                fim_i = url.rfind('/')
                ini_i = url[:fim_i].rfind('/')
                sigla = url[ini_i+1:fim_i]
                if sigla == journal:
                    journals[sigla] = True
                    fw.write(f'{sigla}:{journal_name}\n')
                    break
        if not journals[journal]:
            print(f'\t{journal} ainda nao encontrado')
        else:
            print(f'\t{journal} encontrado')

fw.close()