import pandas as pd
import numpy as np
import time
from sklearn.feature_extraction.text import CountVectorizer
import operator
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import string
from sklearn.cross_validation import train_test_split

def rmse(p, y):
    diff = p - y
    acc = np.sum(np.square(diff))
    acc = acc/float(p.shape[0])
    return acc**(1/(float(2)))

#Index([u'critic', u'fresh', u'imdb', u'link', u'publication', u'quote', u'review_date', u'rtid', u'title'], dtype=object)
def mkdf():
    reviews = pd.read_csv('reviews.csv')
    reviews = reviews[~reviews.quote.isnull()]
    reviews = reviews[reviews.fresh != 'none']
    reviews = reviews[reviews.quote.str.len() > 0]
    return reviews

"""
Find the 30 critics with the most reviews, and list their names in a table along with
   (a) the name of the publication they work for
   (b) the date of their first review
   (c) the date of their last review

shout out to http://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
"""
def top30critics(reviews):
    start_time = time.time()
    #    reviews = reviews[['publication','rtid','review_date','critic']]
    rbyc = reviews.groupby(['critic'])
    aggregrate = {
        'rtid' : 'count',
        'publication' : lambda x : ', '.join(x.unique()),
        'review_date' : {
            'first' : 'min',
            'last' : 'max'
        }
    }
    def sanitize(name):
        return "publication(s)" if name == "<lambda>" else name
    new_df = rbyc.agg(aggregrate)
    new_df.columns = [sanitize(c[1]) for c in new_df.columns]
    new_df.sort(['count'], ascending=False, inplace=True)
#    print "top30 took seconds", time.time() - start_time
    print new_df.head(30)

def vectorize(corpus):
    stemmer = PorterStemmer()
    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
            return stemmed

    def tokenize(text):
        #to remove punctuations and
        #general usage of PorterStemmer (so/questions/26126442/)
        text = ''.join([ch for ch in text if ch not in string.punctuation])
        tokens = word_tokenize(text)
        stems = stem_tokens(tokens, stemmer)
        return stems

    vectorizer = CountVectorizer(stop_words='english', binary=True)
    return vectorizer.fit(corpus)

def top(phi_f, phi_r, vectorizer, n=10):
    V = vectorizer.get_feature_names()
    zipped = zip(phi_f, phi_r)
    relevant_fresh = []
    relevant_rotten = []
    i = 0
    for pair in zipped:
        p_f = pair[0]
        p_r = pair[1]
        if p_f == 0.0 or p_r == 0.0:
            relevant_fresh.append((0.0, i))
            relevant_rotten.append((0.0, i))
        else:
            relevant_fresh.append((np.log(p_f/p_r),i))
            relevant_rotten.append((np.log(p_r/p_f),i))
        i += 1

    print "top fresh indicators"
    n1 = n
    for p in sorted(relevant_fresh, key=operator.itemgetter(0), reverse=True):
        n1 -= 1
        if n1 < 0:
            break
        print p[0], V[p[1]]

    print "top rotten indicators"
    n1 = n
    for p in sorted(relevant_rotten, key=operator.itemgetter(0), reverse=True):
        n1 -= 1
        if n1 < 0:
            break
        print p[0], V[p[1]]

def mknb(vectorizer, X, y):
    nreviews = len(X)
    f = np.array([1 if s == 'fresh' else 0 for s in y])
    nfresh = np.sum(f)
    X = vectorizer.transform(X).toarray()

    #m * |V| matrix is returned above
    nfresh = float(nfresh)
    nreviews = float(nreviews)
    phi_word_fresh = (np.dot(f,X) + 1)/(nfresh + 2)
    phi_word_rotten = (np.dot(1-f,X) + 1)/(nreviews - nfresh + 2)
    phi_fresh = nfresh/nreviews
    #top(phi_word_fresh, phi_word_rotten, vectorizer, n=15)
    """
    p(y=1|x) = p(x|y=1)*p(y=1)  =       p(x|y=1)*p(y=1)
               ------------     -------------------------
                   p(x)          p(x|y=1)*p(y=1)+p(x|y=0)*p(y=0)

    p(x|y=1) -> *p(x_i|y=1) for each word i
    """

    def prob_nb_assumption(x, phi):
        """To get p(x|y), take a element-wise product between the phi vector
        and the vectorized form of the quote. This will give a row vector
        per quote of the form [.5, 0, .3, 0, 0 ...] .. To calculate the
        actual probability, we have to avoid the 0s and take log() of the
        the other values.. Then sum them up and take exp() on it to get
        the orginally intended project of independent probabilties"""
        vfunc = np.vectorize(lambda r : (0.0 if r == 0 else np.log(r)))
        prob = np.multiply(phi, x)
        return np.exp(np.sum(vfunc(prob),axis=1))

    def classifier(quotes):
        Xquotes = vectorizer.transform(quotes).toarray()
        fresh_prob = prob_nb_assumption(Xquotes, phi_word_fresh)*phi_fresh
        print fresh_prob
        rotten_prob = prob_nb_assumption(Xquotes, phi_word_rotten)*(1-phi_fresh)
        prob = fresh_prob/(fresh_prob + rotten_prob)
        return np.array(['fresh' if p >= .5 else 'rotten' for p in prob])

    return classifier

def xmain():
    reviews = mkdf()
    #top30critics(reviews)

    #assumption: since we do not want to see a word in test
    #not seen during train, consider words in the whole corpus
    #before training
    vectorizer = vectorize(reviews['quote'])
    train_X, test_X, train_y, test_y = train_test_split(reviews['quote'],
                                                        reviews['fresh'],
                                                        test_size=.33, random_state=7)
    #reviews[:.70*len(reviews)]
    #test = reviews[.70*len(reviews):]
    classifier = mknb(vectorizer, train_X, train_y) #returns a functor
    pred = classifier(test_X) #returns a series
    print np.sum(pred == test_y)/float(len(test_y))

if __name__ == "__main__":
    xmain()
            
#    x= reviews[0:30]['fresh'] == 'fresh'
#    y= reviews[100:130]['fresh'] == 'fresh'
#    print rmse(x.values, y.values)
