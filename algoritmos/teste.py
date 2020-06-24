import numpy as np
import xml.etree.ElementTree as ET
import pickle

arq_o = 'G:\\Mestrado\\BD\\data\\dblp2.xml'

dblp = open(arq_o, 'r', encoding="utf-8")
root = ET.parse(dblp).getroot()
dblp.close()
journals = {}
journals_publications = {}
set_of_authors = set()

artigos = 1
for child in root:
    if(child.tag == "article"):
        authors = []
        for attr in child:
            if(attr.tag == "author"):
                if attr.text == "Michael Stonebraker":
                    print("Artigo #{}".format(artigos))
                    artigos +=1

            if(attr.tag == "journal"):
                print(attr.text)