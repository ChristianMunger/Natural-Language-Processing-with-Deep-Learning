from __future__ import print_function, division
from future.utils import iteritems
from builtins import range

from gensim.models import KeyedVectors

#3 million words and phrases. Most people have a vocab of 20000. 
#Phrases take up a lot of this dataset, New York == New_York
word_vectors = KeyedVectors.load_word2vec_format(
  r'C:\Users\chris\Desktop\NLP Course\GoogleNews-vectors-negative300.bin.gz',
  binary=True
)

def nearest_neighbors(w):
    r = word_vectors.most_similar(positive=[w])
    print("neighbors of %s" % w)
    for word, score in r:
        print("\t%s" % word)


nearest_neighbors("Tesla")

