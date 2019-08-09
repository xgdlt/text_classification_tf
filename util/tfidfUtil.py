#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from sklearn.feature_extraction.text import TfidfVectorizer

def get_word_to_idf(examples, filePath=None):
    model = TfidfVectorizer( max_df=0.5, min_df=3).fit(examples)
    word2idf = {}
    for word, index in model.vocabulary_.items():
        word2idf[word] = model.idf_[index]
    if filePath:
        fw = open(filePath,"w")
        for word, idf in word2idf.items():
            fw.write("%s\t%.4f"%(word,idf))
    return word2idf

def get_module(word2idf):
    return math.sqrt(sum([ v*v for k,v in word2idf.items()]))

def get_words_modul(words,word2idf):
    return math.sqrt(sum([ word2idf.get(word,0)*word2idf.get(word,0) for word in words]))

def cover_to_word_to_idf(words,word2idfs, default_value=0.0):
    token_word2idf = {}
    for word in word2idfs:
        token_word2idf[word] = token_word2idf.get(word,0) + word2idfs.get(word,default_value)

def get_similar_ratio(token_a, token_b, word2idf):
    token_a_words = token_a.split()
    token_b_words = token_b.split()
    module_a = get_words_modul(token_a_words, word2idf)
    module_b = get_words_modul(token_b_words, word2idf)
    if module_a ==0 or module_b==0:
        return 0.0
    sum=0
    for word, idf_value in token_a_words:
        if word not in token_b_words: continue
        sum += idf_value*token_b_words[word]
    return sum/(module_a*module_b)

def get_most_similar_example(token, examples, word2idf=None):
    if not word2idf:
        word2idf = get_word_to_idf(examples)
    max_ratio = 0.0
    max_ratio_example_index = -1;
    for index,example in enumerate(examples):
        ratio = get_similar_ratio(token, example)
        if max_ratio < ratio:
            max_ratio = ratio
            max_ratio_example_index = index
    return max_ratio, max_ratio_example_index