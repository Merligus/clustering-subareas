import html
import codecs

arq_i = '../data/dblp.xml'
arq_o = '../data/dblp2.xml'

# to count the number of lines open the dblp before
dblp = codecs.open(arq_i, encoding="iso-8859-1")
count = 0
for line in dblp:
    count += 1
dblp.close()
dblp2 = open(arq_o, "w", encoding="utf-8")

dblp = codecs.open(arq_i, encoding="iso-8859-1")
for i, line in enumerate(dblp):
    res = line.replace("&lt;", "")
    res = res.replace("&#60;", "")
    res = res.replace("&gt;", "")
    res = res.replace("&#62;", "")
    res = res.replace("&amp;", "and")
    res = res.replace("&#38;", "and")
    res = res.replace("&quot;", "")
    res = res.replace("&#34;", "")
    res = res.replace("&apos;", "")
    res = res.replace("&#39;", "")
    dblp2.write(html.unescape(res))
    
    if i % 1000 == 0:
        print(f"Line {i} of {count}")

dblp.close()
dblp2.close()
