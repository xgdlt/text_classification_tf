#!usr/bin/env python
# coding:utf-8
"""
Author:
    LiTeng 1471356861@qq.com

"""

import tensorflow as tf
from tensorflow import keras


class model(keras.Model):

    def __init__(self,  config):
        super(model, self).__init__()
        self.config = config
        self.fcs = []
        for i in range(0, config.MLP.layer_num - 1):
            self.fcs.append( keras.layers.Dense(config.MLP.hiden_dimensions[i], activation = 'relu',kernel_regularizer=keras.regularizers.L1L2(0.01,0.01)))
        self.flatten = keras.layers.Flatten()
        #self.dropout = keras.layers.Dropout(config.MLP.dropout)
        self.fc = keras.layers.Dense(config.MLP.num_classes,kernel_regularizer=keras.regularizers.L1L2(0.01,0.01))
        self.regression = keras.layers.Dense(1, name = "regression",kernel_regularizer=keras.regularizers.L1L2(0.01,0.01))
        self.classify = keras.layers.Dense(config.MLP.num_classes, name = "classify",kernel_regularizer=keras.regularizers.L1L2(0.01,0.01))

    def call(self, inputs,training=None, mask=None):
        print("inputs",inputs )

        if self.config.rest:
            out, base = inputs
        else:
            out = inputs

        for i in range(0, len(self.fcs)):
            out =  self.fcs[i](out)
            print("fc %d"%i, out)

        out = self.flatten(out)
        print("out", out)

        if self.config.rest:
            out = keras.layers.concatenate([out, base])

        regression = self.regression(out)
        if self.config.MLP.classify:
            classify = self.classify(out)
            classify = tf.nn.softmax(classify)
            return [regression, classify]
        return regression