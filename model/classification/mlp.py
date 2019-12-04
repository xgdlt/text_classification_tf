#!usr/bin/env python
# coding:utf-8
"""
Author:
    LiTeng 1471356861@qq.com

"""

import tensorflow as tf
from tensorflow import keras


class MLP(keras.Model):

    def __init__(self,  config):
        super(MLP, self).__init__()
        self.config = config
        self.fcs = []
        for i in range(0, config.MLP.layer_num - 1):
            self.fcs.append( keras.layers.Dense(config.MLP.hiden_dimensions[i]), activation = 'relu')

        self.dropout = keras.layers.Dropout(config.MLP.dropout)
        self.fc = keras.layers.Dense(config.MLP.num_classes)


    def call(self, inputs,training=None, mask=None):
        print("inputs",inputs )
        out = inputs
        for i in range(0, len(self.fcs)):
            out =  self.fcs[i](out)
            out = self.dropout(out)
        out = self.fc(out)
        print("fc", out)

        if self.config.logits_type == "softmax":
            out = tf.nn.softmax(out)
        elif self.config.logits_type == "sigmoid":
            out = tf.nn.sigmoid(out)

        return  out