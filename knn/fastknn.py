# --------------------------------------实例一-----------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from util import tfidfUtil

class fastknn(object):
    def __init__(self, examples, labels):
        self.examples = examples
        self.labels = labels
        self.word2idfs = tfidfUtil.get_word_to_idf(examples)

    def predict(self,example):
        ratio,index = tfidfUtil.get_most_similar_example(example, self.examples, self.word2idfs)
        return ratio,self.examples[index],self.labels[index]
