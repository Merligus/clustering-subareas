from collections import defaultdict
import nltk
import string

# n = top n
def show_top(sentences, frequent, bad_words, file, n=10):
    length = len(sentences)
    for s1 in sentences:
        token1 = nltk.word_tokenize(s1)
        set_token1 = set(token1)
        for t in set_token1:
            if t not in string.punctuation:
                if t not in bad_words:
                    if not t.isnumeric():
                        frequent[t] += 1
    top = sorted(frequent.items(), key=lambda item: item[1], reverse=True)[:n]
    for word, _ in top:
        print(f'\t\t{word} {frequent[word]/length:.0%}')
        file.write(f'\t\t{word} {frequent[word]/length:.0%}\n')

bad_words = {'cambridge', 'cambridge', 'ca', 'california', 'jose', 'int', 'second', 'third', '4th', '5th', '6th', 
             '7th', '8th', '9th', '10th', 'san', 'information', 'first', 'va', '3rd', '2nd', 'de', '2010.', 'annual', 'systems', 'selected', 'revised', 'symposium', 'papers', 'acm',
             'comput', 'computational', 'ieee', 'j.', 'computing', 'international', 'proceeding',
             'proceedings', 'of', 'to', 'for', 'comp.', 'computer', 'conference', 'workshop',
             'on', 'and', 'or', 'the', 'that', 'this', 'with', 'good', 'bad', 'show', 'in',
             'january', 'february', 'march', 'may', 'april', 'june', 'july', 'august', 'october',
             'september', 'november', 'december', 'journal'}

comb = []
for cut in [0, 0.2]:
    for only_journals in [0, 1]:
        for year in [0, 2010]:
            if only_journals and cut > 0:
                continue
            in_name = ""
            if year > 0:
                in_name += '_' + str(year)
            if only_journals:
                in_name += '_only_journals'
            if cut > 0:
                in_name += f'_cut{cut:.3}'
            comb.append(in_name)

for in_name in comb:
    for function, rec in [('multilevel', 3)]:
        for mode in ['max', 'union', 'mean']:
            dir = "G:\\Mestrado\\BD\\data\\formatted_output\\definitivas\\"
            file_name = f"{function}_{mode}_rec{rec}{in_name}.txt"
            f_out = open(f'{dir}most_frequent_words\\mfw_{file_name}', 'w', encoding="utf-8")

            frequent = defaultdict(int)
            sentences = []
            with open(dir + file_name, "r", encoding="utf-8") as f:
                for line in f:
                    if line[0].isnumeric():
                        if len(sentences) > 0:
                            print(f'\tWith size {len(sentences)}')
                            f_out.write(f'\tWith size {len(sentences)}\n')
                            show_top(sentences, frequent, bad_words, f_out)

                        frequent = defaultdict(int)
                        sentences = []
                        print(f'Processing cluster {line[:-1]}')
                        f_out.write(f'Processing cluster {line[:-1]}\n')
                    elif line[0] == '\t' and line[1].isnumeric():
                        index1 = line.find(':')
                        index2 = line[index1+1:].find(':')
                        sentences.append(line[index1 + index2 + 2:].lower())

                show_top(sentences, frequent, bad_words, f_out)

            f_out.close()