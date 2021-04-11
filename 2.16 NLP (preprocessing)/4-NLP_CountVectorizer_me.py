#Import Libraries
from sklearn.cluster import DBSCAN

from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import v_measure_score , accuracy_score
# ----------------------------------------------------
# Perform DBSCAN clustering from vector array or distance matrix.
# DBSCAN - Density-Based Spatial Clustering of Applications with Noise. 
# Finds core samples of high density and expands clusters from them. 
# Good for data which contains clusters of similar density.

'''
sklearn.feature_extraction.text.CountVectorizer(input='content', encoding='utf-8', decode_error='strict', strip_accents=None, 
                                                    lowercase=True, preprocessor=None, tokenizer=None, stop_words=None, 
                                                    token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), analyzer='word', 
                                                    max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, 
                                                    dtype=<class 'numpy.int64'>)

=======
    - input string {‘filename’, ‘file’, ‘content’}, default=’content’
    - encoding string, default=’utf-8’
    - decode_error {‘strict’, ‘ignore’, ‘replace’}, default=’strict’
    - strip_accents {‘ascii’, ‘unicode’}, default=None
    - lowercase bool, default=True
    - preprocessor callable, default=None
    - tokenizer callable, default=None
    - stop_words string {‘english’}, list, default=None
    - token_pattern str, default=r”(?u)\b\w\w+\b”
    - ngram_range tuple (min_n, max_n), default=(1, 1)
    - analyzer {‘word’, ‘char’, ‘char_wb’} or callable, default=’word’
    - max_df float in range [0.0, 1.0] or int, default=1.0
    - min_df float in range [0.0, 1.0] or int, default=1
    - max_features int, default=None
    - vocabulary Mapping or iterable, default=None
    - binary bool, default=False
    - dtype type, default=np.int64
=======

'''
# ----------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'This is the first document. document document',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
# --------------------
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(corpus)

# print(vectorizer.get_feature_names())
# print("="*25)

# print(X)
# print("="*5)

# print(X.toarray())
# print("="*25)

# ----------------------------------------------------
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
X2 = vectorizer2.fit_transform(corpus)
print(vectorizer2.get_feature_names())
print("="*25)
# ----------------------------------------------------
