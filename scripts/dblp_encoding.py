import html
import codecs

arq_i = 'G:\\Mestrado\\BD\\data\\dblp_new.xml'
arq_o = 'G:\\Mestrado\\BD\\data\\dblp_new2.xml'

dblp = codecs.open(arq_i, encoding="iso-8859-1")
dblp2 = open(arq_o, "w", encoding="utf-8")
for line in dblp:
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
dblp.close()
dblp2.close()