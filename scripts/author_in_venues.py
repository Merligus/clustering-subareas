import requests
from collections import defaultdict

venues = {"scc", "w2gis"}
authors = ["Christophe Claramunt","Andrea Ballatore","Kai-Florian Richter"]
authors_d = defaultdict(list)
for author in authors:
    params = {'q' : author, 'format' : 'json', 'h' : '1000', 'c' : '0'}
    try:
        jason_object = requests.get('https://dblp.org/search/publ/api', params=params).json()
    except:
        print(f'nao encontrado {author}')

    if 'result' in jason_object:
        if 'hits' in jason_object['result']:
            if 'hit' not in jason_object['result']['hits']:
                continue
        else:
            continue
    else:
        continue
    for hit in jason_object['result']['hits']['hit']:
        url = hit['info']['url']
        for venue in venues:
            if url.find(venue) > -1:
                if venue not in authors_d[author]:
                    authors_d[author].append(venue)

for author in authors:
    for venue in venues:
        if venue not in authors_d[author]:
            print(f'{author} not in {venue}')