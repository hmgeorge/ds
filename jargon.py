import pandas as pd
import numpy as np
import ngram
import itertools
import math
import time
from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt 

def dendro(jd_matrix):
    # Dictionary conatining the groupnames 
    group_names = {1 :'Eco & Evol', 2 :'Mole & Cell Bio',3 :'Econ',4 :'Sociology',5 :'Prob & Stats',
                   6 :'Org & mktng',7 :'Law', 8 :'Anthro',9 :'Polit Sci',10 :'Edu'}
    # Plotting the dendrogram
    # Creating a linkage matrix using average distance
    Z = linkage(jd_matrix, 'average')
    plt.figure(figsize = (16,10))
    plt.title("Dendrogram of distance between groups")
    plt.ylabel("Jargon distance")
    #dn = dendrogram(Z, labels = range(1,11), above_threshold_color = 'r')

    # function to provide lables to the leaves of the dendrogram
    def llf(idx):
        return group_names[idx + 1]

    # Creating the dendrogram
    dn = dendrogram(Z, leaf_label_func=llf, leaf_rotation=90, leaf_font_size=8.)
    plt.show()

def normalize_dist(d):
    Z = sum(d.values())
    for k in d:
        d[k] = (1.0*d[k])/float(Z)
    return d

def teleport(x_dist, corpus_dist, alpha=.01):
    all_keys=corpus_dist.keys()
    px = dict()
    for k in all_keys:
        px[k] = alpha*corpus_dist[k]
        if k in x_dist:
            px[k] += (1-alpha)*x_dist[k]

    return px

#make a dict having all possible ngrams and their normalized counts
def mkcorpus(all_ngrams):
    corpus = dict()
    for d in all_ngrams: #take each doc
        for ngram in all_ngrams[d]: #take each ngram for this doc
            n=ngram[0]
            c=ngram[1]
            if n not in corpus:
                corpus[n] = 0
            corpus[n] += c
    return normalize_dist(corpus)

def mkcodebook(all_ngrams, doc_list):
    px = dict()
    for d in doc_list:
        ngram = all_ngrams[d]
        for n in ngram:
            if n[0] not in px:
                px[n[0]] = 0
            px[n[0]] += n[1]

    return normalize_dist(px)

def read_abstracts(abstracts_file):
    # Importing the abstracts text file
    abstract = pd.read_table(abstracts_file, header = None)
    # Renaming the columns
    abstract.columns = ['DocID', 'Text']

    # Removing the rows which has text has null
    abstract = abstract[abstract['Text'] != "null"]
    df = abstract[abstract['Text'] != "abstract"]

    return df

def read_stopwords():
    # Importing stopwords text file
    f = open("stopwords.txt", 'r')

    l=f.readlines()[0].replace('"','')
    stop_words=set(l.split(','))

    return stop_words

def read_groups(groups_file):
    groups = dict()
    with open(groups_file) as f:
        first_line = 1;
        for l in f:
            if first_line:
                first_line = 0;
                continue
            l_words = l.split('\t')
            if len(l_words) < 2:
                continue
            groups[str(l_words[0])] = l_words[1].strip(' \n')
    f.close()
    return groups

#this returns the shannon entropy if pi == pj
#and cross entropy otherwise
#assume all keys in pi are present in pj
#use teleport to create a merged codebook if needed
def derive_entropy(pi, pj):
    Z=0.0
    for x in pi:
        Z += pi[x]*math.log(pj[x],2)
    return -Z

def xmain(abstracts_file="abstracts2.txt", groups_file="groups2.txt"):
    now=time.time()
    df = read_abstracts(abstracts_file)
    stop_words = read_stopwords()
    groups = read_groups(groups_file)

    df['Text'] = df.apply(lambda row: ngram.mod_text(row['Text'], stop_words), axis=1)
    df['DocID'] = df.apply(lambda row: str(row['DocID']), axis=1)

    docs_per_group = dict()
    for k, g in itertools.groupby(df['DocID'], lambda d : groups[d]):
        if k not in docs_per_group:
            docs_per_group[k] = []
        [docs_per_group[k].append(i) for i in g]

    all_1grams = dict()
    for r in df.iterrows():
        doc = str(r[1]['DocID'])
        text = r[1]['Text']
        all_1grams[doc] = ngram.mkngram(text, 1)
    #print all_1grams

    corpus = mkcorpus(all_1grams) #[1]
    #print corpus

    codebook = dict()
    for d in docs_per_group:
        #print d, docs_per_group[d]
        codebook[d] = mkcodebook(all_1grams, docs_per_group[d]) #[2]
    #print codebook

    entropy_matrix = pd.DataFrame(index=sorted(docs_per_group.keys(), key=lambda x : int(x)),
                                  columns=sorted(docs_per_group.keys(), key=lambda x : int(x)))
    merged_codebook = dict()
    for c in codebook:
        merged_codebook[c] = teleport(codebook[c], corpus)

    for i in merged_codebook:
        for j in merged_codebook:
            entropy_matrix[j][i] = derive_entropy(merged_codebook[i], merged_codebook[j])

    print "time taken %f seconds" % (time.time()-now)
    print entropy_matrix

def vectorize(group_ngram, corpus_keys):
    dist = np.zeros(len(corpus_keys))
    for i in range(len(corpus_keys)):
        ck = corpus_keys[i]
        if ck in group_ngram:
            dist[i] += group_ngram[ck]
    return dist

#v2 using numpy
#13.8 seconds on abstracts2 with 1gram
#33.4 seconds on abstracts2 with 3gram
def xmain_np(abstracts_file="abstracts2.txt", groups_file="groups2.txt", ngram_count=1):
    now=time.time()
    df = read_abstracts(abstracts_file)
    stop_words = read_stopwords()
    groups = read_groups(groups_file)

    df['Text'] = df.apply(lambda row: ngram.mod_text(row['Text'], stop_words), axis=1)
    df['DocID'] = df.apply(lambda row: str(row['DocID']), axis=1)

    docs_per_group = dict()
    for k, g in itertools.groupby(df['DocID'], lambda d : groups[d]):
        if k not in docs_per_group:
            docs_per_group[k] = []
        [docs_per_group[k].append(i) for i in g]

    all_ngrams = dict()
    for r in df.iterrows():
        doc = str(r[1]['DocID'])
        text = r[1]['Text']
        all_ngrams[doc] = ngram.mkngram(text, ngram_count)
    #print all_ngrams

    corpus = mkcorpus(all_ngrams)
    corpus_keys = sorted(corpus.keys())
    corpus_vector = vectorize(corpus, corpus_keys)

    alpha=.01
    codebook = dict()
    for g in docs_per_group:
        cb = mkcodebook(all_ngrams, docs_per_group[g])
        codebook[g] = (1-alpha)*vectorize(cb, corpus_keys) + alpha*corpus_vector
    #print codebook

    entropy_matrix = pd.DataFrame(index=sorted(docs_per_group.keys(), key=lambda x : int(x)),
                                  columns=sorted(docs_per_group.keys(), key=lambda x : int(x)))
    
    #this returns the shannon entropy if pi == pj
    #and cross entropy otherwise
    #assume all keys in pi are present in pj
    for i in codebook:
        for j in codebook:
            entropy_matrix[j][i] = -np.dot(codebook[i],np.log2(codebook[j]))

    jd_matrix = np.array(np.zeros(shape=(len(docs_per_group.keys()),len(docs_per_group.keys()))))
    for i in docs_per_group:
        for j in docs_per_group:
            if i == j:
                jd_matrix[int(i)-1][int(j)-1] = 0.0
            else:
                jd_matrix[int(j)-1][int(i)-1] = 1 - entropy_matrix[i][i]/entropy_matrix[i][j]
                #jd_matrix[int(i)-1][int(j)-1] = 1 - entropy_matrix[i][i]/entropy_matrix[i][j]

    #list(np.diagonal(entropy_matrix.values)) returns diag elements in entropy_matrix
    print "time taken %f seconds" % (time.time()-now)
    dendro(jd_matrix)
    print entropy_matrix

def xmain_cv(abstracts_file="abstracts2.txt", groups_file="groups2.txt", ngram_count=1):
    now=time.time()
    df = read_abstracts(abstracts_file)
    stop_words = read_stopwords()
    groups = read_groups(groups_file)

    df['Text'] = df.apply(lambda row: ngram.mod_text(row['Text'], stop_words), axis=1)
    df['DocID'] = df.apply(lambda row: str(row['DocID']), axis=1)

    docs_per_group = dict()
    for k, g in itertools.groupby(df['DocID'], lambda d : groups[d]):
        if k not in docs_per_group:
            docs_per_group[k] = []
        [docs_per_group[k].append(i) for i in g]

    vectorizer = CountVectorizer(min_df = 0)
    X = vectorizer.fit_transform(df['Text']).toarray()
    groups_raw = dict()
    for r in df.iterrows():
        docid = r[1]['DocID']
        text = r[1]['Text']
        group = groups[docid]
        if group not in groups_raw:
            groups_raw[group] = []
        groups_raw[group].append(text)

    corpus_vector = X.sum(axis=0)
    corpus_vector = corpus_vector/float(corpus_vector.sum())

    alpha=.01
    codebook = dict()
    for g in docs_per_group:
        X = vectorizer.transform(groups_raw[g]).toarray()
        dist_vector = X.sum(axis=0)
        codebook[g] = dist_vector/float(dist_vector.sum())

    for i in codebook:
        e = 0.0
        for j in codebook[i]:
            if j != 0:
                e += j*np.log2(j)
        print -e

    entropy_matrix = pd.DataFrame(index=sorted(docs_per_group.keys(), key=lambda x : int(x)),
                                  columns=sorted(docs_per_group.keys(), key=lambda x : int(x)))
    
    for i in codebook:
        codebook[i] = (1-alpha)*codebook[i] + alpha*corpus_vector

    #this returns the shannon entropy if pi == pj
    #and cross entropy otherwise
    #assume all keys in pi are present in pj
    for i in codebook:
        for j in codebook:
            entropy_matrix[j][i] = -np.dot(codebook[i],np.log2(codebook[j]))
#            e = 0.0;
#            for k in range(len(codebook[i])):
#                if codebook[j][k] != 0:
#                    e += codebook[i][k]*np.log2(codebook[j][k])
#            entropy_matrix[j][i] = -e;

    jd_matrix = np.array(np.zeros(shape=(len(docs_per_group.keys()),len(docs_per_group.keys()))))
    for i in docs_per_group:
        for j in docs_per_group:
            if i == j:
                jd_matrix[int(i)-1][int(j)-1] = 0.0
            else:
                jd_matrix[int(j)-1][int(i)-1] = 1 - entropy_matrix[i][i]/entropy_matrix[i][j]
                #jd_matrix[int(i)-1][int(j)-1] = 1 - entropy_matrix[i][i]/entropy_matrix[i][j]

    #list(np.diagonal(entropy_matrix.values)) returns diag elements in entropy_matrix
    print "time taken %f seconds" % (time.time()-now)
    dendro(jd_matrix)
    print entropy_matrix

if __name__ == "__main__":
    xmain_cv()
