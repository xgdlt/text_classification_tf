#!usr/bin/env python
# coding:utf-8
# @time     :2019/10/29 12:00
# @author   :liteng
# @function :tensorflow2.0 for dcnn
# @paper:    A Convolutional Neural Network for Modelling Sentences(http://www.aclweb.org/anthology/P14-1062)

import tensorflow as tf
from tensorflow import keras
from model_tf.layers.layers import wide_convolution,dynamic_k_max_pooling,prem_fold
from model_tf.model_util import select_k


class BasicConvBlock(keras.layers.Layer):

    def __init__(self, input_len_max, filters=64, kernel_sizes=[6,4,3]):
        super(BasicConvBlock, self).__init__()
        self.convs = []
        self.top_k_poolings = []
        self.prem_fold = prem_fold()
        for i,kernel_size in enumerate(kernel_sizes):
            conv = wide_convolution(filters=filters, kernel_size=kernel_size)
            self.convs.append(conv)
            top_k = select_k(input_len_max, len(kernel_sizes), i+1)  # 求取k
            top_k_pooling = dynamic_k_max_pooling(top_k=top_k)
            self.top_k_poolings.append(top_k_pooling)

    def call(self, inputs, training=None):
        out = inputs
        print("inputs", inputs)
        for i in range(len(self.convs)):
            conv = self.convs[i](out)
            print("BasicConvBlock_convs_%d"%i, out)
            if i == len(self.convs )-1 :
                conv = self.prem_fold(conv)
                print("BasicConvBlock_prem_fold", out)
            pool = self.top_k_poolings[i](conv)
            print("BasicConvBlock_top_k_poolings_%d" % i, out)
            out = pool
        return out


class TextDCNN(tf.keras.Model):
    def __init__(self, config):
        super(TextDCNN, self).__init__( )

        self.embedding = keras.layers.Embedding(config.TextDCNN.input_dim, config.TextDCNN.embedding_dimension,
                                                input_length=config.TextDCNN.input_length)
        self.multil_kernel_sizes = config.TextDCNN.kernel_sizes
        self.convs = []

        self.config = config

        for kernel_sizes in self.multil_kernel_sizes:
            conv = BasicConvBlock(input_len_max=config.TextDCNN.input_length, filters=config.TextDCNN.filters, kernel_sizes=kernel_sizes)
            self.convs.append(conv)

        #self.top_k = self.config.TextCNN.top_k_max_pooling
        self.flatten = keras.layers.Flatten()
        self.dropout = keras.layers.Dropout(config.TextDCNN.dropout)
        self.fc = keras.layers.Dense(config.TextCNN.num_classes)

    def call(self, inputs,training=None, mask=None):
        print("inputs", inputs)
        embedding = self.embedding(inputs)
        print("embedding", embedding)
        cnns = []
        for i in range(len(self.convs)):
            x = self.convs[i](embedding)
            cnns.append(x)
            print("conv %d"%i, x)

        x = keras.layers.concatenate(cnns)
        print("concat", x)
        x = self.flatten(x)
        print("flatten ", x)
        x = self.fc(x)
        if self.config.logits_type == "softmax":
            x = tf.nn.softmax(x)
        elif self.config.logits_type == "sigmoid":
            x = tf.nn.sigmoid(x)
        return x