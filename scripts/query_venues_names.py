import requests

## SCRIPT PARA PEGAR OS NOMES DOS VEÍCULOS PELA API DO DBLP


fw = open('G:\\Mestrado\\BD\\data\\journal_names.txt', 'w', encoding="utf-8")
journals = {}
query = 'journal$|conference'
for page in range(8):
    params = {'q' : query, 'format' : 'json', 'h' : '1000', 'c' : '0', 'f': str(page*1000)}
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

        find_c = url.find('/conf/')
        find_j = url.find('/journals/')
        suf = '_'
        suf += 'c' if find_c > 0 else ''
        suf += 'j' if find_j > 0 else ''
        sigla += suf

        if find_c > 0 or find_j > 0:
            if sigla not in journals:
                journals[sigla] = True
                fw.write(f'{sigla}:{journal_name}\n')
fw.close()