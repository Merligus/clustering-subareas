
arq_o = '../data/dblp2.xml'

def show_line(n, before=0, after=10):
    print(50*'*')
    min = n-before
    max = n+after
    with open(arq_o, 'r') as dblp:
        for line_number, line in enumerate(dblp):
            if line_number >= min and line_number <= max:
                print(line[:-1])
            elif line_number > n + after:
                break
    print(50*'*')

show_line_number = -1
sentences = {"<inproceedings": True, "publtype": False}
with open(arq_o, 'r') as dblp:
    for line_number, line in enumerate(dblp):
        all_in = True
        for s in sentences:
            if (s in line) != sentences[s]:
                all_in = False
                break
        if all_in:
            show_line(line_number)
        elif line_number == show_line_number:
            show_line(line_number)