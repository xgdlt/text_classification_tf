#!/usr/bin/env python
# coding:utf-8
"""
Author:
    LiTeng 1471356861@qq.com

Implement TextRCNN  CNN + RNN
"""

import tensorflow as tf
from tensorflow import keras

from model.layers.embeddings import EmbeddingsLayer
from utils.logger import Type


class RNNType(Type):
    RNN = 'RNN'
    LSTM = 'LSTM'
    GRU = 'GRU'

    @classmethod
    def str(cls):
        return ",".join([cls.RNN, cls.LSTM, cls.GRU])

class Model(tf.keras.Model):
    """
    One layer rnn.
    """
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        if self.config.embedding.use_embedding:
            self.embedding = EmbeddingsLayer(config.embedding)
        else:
            self.reshape = keras.layers.Reshape((config.TextRCNN.input_length, config.TextRCNN.embedding_dimension))

        if self.config.TextRNN.rnn_type == RNNType.LSTM:
            layer_cell = keras.layers.LSTM
        elif self.config.TextRNN.rnn_type == RNNType.GRU:
            layer_cell = keras.layers.GRU
        else:
            layer_cell = keras.layers.SimpleRNN

        self.layer_cells = []
        for i in range(config.TextRNN.num_layers):
            if config.TextRNN.bidirectional:
                self.layer_cells.append(keras.layers.Bidirectional(
                    layer_cell(config.TextRNN.hidden_dimension,
                               use_bias=config.TextRNN.use_bias,
                               activation=config.TextRNN.activation,
                               kernel_regularizer=keras.regularizers.l2(self.config.TextRNN.l2 * 0.1),
                               recurrent_regularizer=keras.regularizers.l2(self.config.TextRNN.l2))))
            else:
                self.layer_cells.append(layer_cell(config.TextRNN.hidden_dimension,
                                                   use_bias=config.TextRNN.use_bias,
                                                   activation=config.TextRNN.activation,
                                                   kernel_regularizer=keras.regularizers.l2(
                                                       self.config.TextRNN.l2 * 0.1),
                                                   recurrent_regularizer=keras.regularizers.l2(self.config.TextRNN.l2)))
        if config.TextRNN.bidirectional:
            rnn_out_dimension = config.TextRNN.hidden_dimension * 2
        else:
            rnn_out_dimension = config.TextRNN.hidden_dimension
        self.reshape = keras.layers.Reshape((rnn_out_dimension, 1, 1))

        self.kernel_sizes = config.TextCNN.kernel_sizes
        self.convs = []
        self.pools = []

        for kernel_size, filter_size in zip(config.TextCNN.kernel_sizes, config.TextCNN.filter_sizes):
            conv = keras.layers.Conv2D(filters=filter_size, kernel_size=(kernel_size, 1),
                                       strides=1, padding='valid', activation='relu')
            self.convs.append(conv)
            pool = keras.layers.MaxPool2D(pool_size=(config.TextCNN.input_length - kernel_size + 1, 1), padding='valid')
            self.pools.append(pool)

        self.flatten = keras.layers.Flatten()

        self.fc = keras.layers.Dense(config.TextCNN.num_classes)

    #tf.function(input_signature=[tf.TensorSpec([None, 80], tf.float32)])
    def call(self, inputs, training=None, mask=None):

        print('inputs', inputs)
        x = inputs
        if self.config.embedding.use_embedding:
            # [b, sentence len] => [b, sentence len, word embedding]
            x = self.embedding(x)
            print("embedding", x)
        else:
            x = self.reshape(x)
            print("reshape", x)

        for layer_cell in self.layer_cells:
            x = layer_cell(x)
        print('rnn', x)
        x = self.reshape(x)
        print("reshape ", x)
        cnns = []
        for i in range(len(self.convs)):
            conv = self.convs[i](x)
            pool = self.pools[i](conv)
            cnns.append(pool)
            print("conv %d" % i, conv)
            print("pool %d" % i, pool)

        x = keras.layers.concatenate(cnns)
        print("concat", x)
        x = self.flatten(x)
        print("flatten ", x)
        x = self.fc(x)
        if self.config.logits_type == "softmax":
            x = tf.nn.softmax(x)
        elif self.config.logits_type == "sigmoid":
            x = tf.nn.sigmoid(x)
        print("output ", x)
        return x


