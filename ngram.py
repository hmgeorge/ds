import pandas as pd
from nltk.stem.porter import PorterStemmer
import pprint
import re

#remove stop words and stem remaining
def mod_text(row_text, stop_words, stemmer=None):
    s=[]
    for w in row_text.split(' '):
        w = re.sub('[^a-zA-Z]+', '', w).lower()
        if w == '':
            continue
        if w not in stop_words:
            if stemmer is None:
                s.append(w)
            else:
                s.append(stemmer.stem(w)) 

    return ' '.join(s)

map_d = dict();
map_d['doc1'] = [('a', 1), ('b', 1), ('c', 1)]
map_d['doc2'] = [('b', 1), ('c', 1)]
map_d['doc3'] = [('d', 1), ('c', 1), ('a', 2)]

def mkdocmat(map_d):
    def invertor(imap, kv):
        doc = kv[0]
        for wc in kv[1]:
            w = wc[0]
            if w not in imap:
                imap[w] = []
            imap[w].append((doc, wc[1]))
        return imap

    imap = reduce(invertor, map_d.items(), dict())

    df = pd.DataFrame(index=sorted(map_d.keys()),columns=sorted(imap.keys()))
    for k in imap.keys():
        for d in imap[k]:
            df[k][d[0]] = d[1]

    df = df.fillna(value=0)
    return df

def mkngram(text, count=1, out='list'):
    s = dict()
    words = text.split(" ")

    if count > len(words):
        return s #error

    for i in range(len(words)-count+1):
        ngram = ' '.join(words[i:i+count])
        if ngram not in s.keys():
            s[ngram] = 0
        s[ngram] = s[ngram] + 1

    if out == 'list':
        l = []
        [l.append((k,v)) for k, v in s.iteritems()]
        return l
    else:
        return s 

def xmain(filename="abstracts.txt"):
    # Importing the abstracts text file
    abstract = pd.read_table(filename, header = None)
    # Renaming the columns
    abstract.columns = ['DocID', 'Text']

    # Removing the rows which has text has null
    df = abstract[abstract['Text'] != "null"]

    # Displaying the first few rows
    df.head()

    # Importing stopwords text file
    f = open("stopwords.txt", 'r')

    l=f.readlines()[0].replace('"','')
    stop_words=set(l.split(','))

    stemmer = PorterStemmer()
    df['Text'] = df.apply(lambda row: mod_text(row['Text'], stop_words, stemmer), axis=1)

    all_1grams = dict()
    all_2grams = dict()
    for r in df.iterrows():
        doc = str(r[1]['DocID'])
        text = r[1]['Text']
        all_1grams[doc] = mkngram(text, 1)
        all_2grams[doc] = mkngram(text, 2)

    df_1gram = mkdocmat(all_1grams)
    df_2gram = mkdocmat(all_2grams)

    print df_1gram
    print df_2gram
