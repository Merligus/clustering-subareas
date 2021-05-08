from collections import defaultdict
import nltk
import string

def show_top(sentences, frequent, bad_words):
    i = 0
    for s1 in sentences[:-1]:
        token1 = nltk.word_tokenize(s1)
        set_token1 = set(token1)
        for s2 in sentences[i+1:]:
            token2 = nltk.word_tokenize(s2)
            set_token2 = set(token2)
            set_r = set_token1.intersection(set_token2)
            for word in set_r:
                if word not in string.punctuation:
                    if word not in bad_words:
                        if not word.isnumeric():
                            frequent[word] += 1          
        i += 1
    top = sorted(frequent.items(), key=lambda item: item[1], reverse=True)[:n]
    for word, freq in top:
        print(f'\t\t{word} {freq}')

bad_words = {'korea', 'jose', 'china', 'int', 'india', 'second', 'spain', 'poland', 'third', 'singapore', '4th', '5th', '6th', '7th', '8th', '9th', '10th', 'romania', 'california', 'san', 'information', 'first', 'va', '3rd', '2nd', 'finland', 'brazilian', 'de', '2010.', 'annual', 'ca', 'systems', 'france', 'uk', 'brazil', 'usa', 'japan', 'germany', 'selected', 'revised', 'symposium', 'papers', 'acm', 'comput', 'computational', 'ieee', 'j.', 'computing', 'international', 'proceeding', 'proceedings', 'of', 'to', 'for', 'comp.', 'computer', 'conference', 'workshop', 'on', 'and', 'or', 'the', 'that', 'this', 'with', 'good', 'bad', 'show', 'in', 'january', 'february', 'march', 'may', 'april', 'june', 'july', 'august', 'october', 'september', 'november', 'december'}
# top 10
n = 10
dir = "G:\\Mestrado\\BD\\data\\formatted_output\\"
file_name = 'multilevel_union_rec3_.txt'

frequent = defaultdict(int)
sentences = []

with open(dir + file_name, "r") as f:
    for line in f:
        if line[0].isnumeric():
            if len(sentences) > 0:
                print(f'\tWith size {len(sentences)}')
                show_top(sentences, frequent, bad_words)

            frequent = defaultdict(int)
            sentences = []
            print(f'Processing cluster {line[:-1]}')
        elif line[0] == '\t' and line[1].isnumeric():
            index1 = line.find(':')
            index2 = line[index1+1:].find(':')
            sentences.append(line[index1 + index2 + 2:].lower())

    show_top(sentences, frequent, bad_words)