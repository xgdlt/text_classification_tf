#!/usr/bin/env python
# coding:utf-8
"""
Author:
    LiTeng 1471356861@qq.com

Implement RNN, contains LSTM，GRU，RNN
Reference: "Effective LSTMs for Target-Dependent Sentiment Classification"
               "Bidirectional LSTM-CRF Models for Sequence Tagging"
               "Generative and discriminative text classification
                with recurrent neural networks"
"""

import tensorflow as tf
from tensorflow import keras

from utils.logger import Type


class RNNType(Type):
    RNN = 'RNN'
    LSTM = 'LSTM'
    GRU = 'GRU'

    @classmethod
    def str(cls):
        return ",".join([cls.RNN, cls.LSTM, cls.GRU])


class model(tf.keras.Model):
    """
    One layer rnn.
    """
    def __init__(self, config):
        super(model, self).__init__()
        self.config = config
        if  self.config.RNN.rnn_type == RNNType.LSTM:
            layer_cell = keras.layers.LSTM
        elif self.config.RNN.rnn_type == RNNType.GRU:
            layer_cell = keras.layers.GRU
        else:
            layer_cell = keras.layers.SimpleRNN

        self.rnn_type = config.RNN.rnn_type
        self.num_layers = config.RNN.num_layers
        self.bidirectional = config.RNN.bidirectional
        #self.embedding = keras.layers.Embedding(config.RNN.input_dim, config.RNN.embedding_dimension,
        #                                        input_length=config.RNN.input_length)

        self.layer_cells = []
        for i in range(config.RNN.num_layers):
            if config.RNN.bidirectional:
                self.layer_cells.append(keras.layers.Bidirectional(
                     layer_cell(config.RNN.hidden_dimension,
                            use_bias=config.RNN.use_bias,
                            activation=config.RNN.activation,
                            kernel_regularizer=keras.regularizers.l2(self.config.RNN.l2 * 0.1),
                            recurrent_regularizer=keras.regularizers.l2(self.config.RNN.l2))))
            else:
                self.layer_cells.append(layer_cell(config.RNN.hidden_dimension,
                               use_bias=config.RNN.use_bias,
                               activation=config.RNN.activation,
                               kernel_regularizer=keras.regularizers.l2(self.config.RNN.l2 * 0.1),
                               recurrent_regularizer=keras.regularizers.l2(self.config.RNN.l2)))

        self.regression = keras.layers.Dense(1)
        self.classify = keras.layers.Dense(config.MLP.num_classes)

    def call(self, inputs, training=None, mask=None):

        print('inputs', inputs)
        # [b, sentence len] => [b, sentence len, word embedding]
        #x = self.embedding(inputs)
        #print('embedding', x)
        if self.config.rest:
            x,base = inputs
        else:
            x = inputs
        for layer_cell in self.layer_cells:
            x = layer_cell(x)
        print('rnn', x)

        if self.config.rest:
            x = keras.layers.concatenate([x, base])

        regression = self.regression(x)
        if self.config.RNN.classify:
            classify = self.classify(x)
            classify = tf.nn.softmax(classify)
            return [regression, classify]
        return regression


