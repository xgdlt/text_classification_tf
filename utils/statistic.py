#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

class wordStatistic(object):
    def __init__(self, presuffix="word"):
        self.t = {}
        self.presuffix = presuffix

    def add(self, word):
        self.t[word] = self.t.get(word, 0) + 1

    def printCntWithHistory(self):
        f = open("%s2cnt"%self.presuffix, "w")
        denominator = sum([v for k,v in self.t.items()])
        numerator = 0
        for word,cnt in sorted(self.t.items(), key = lambda x:x[0]):
            numerator += cnt
            ratio = numerator / float(denominator)
            f.write("%s\t%d\t%.4f\n"%(str(word), cnt, ratio))