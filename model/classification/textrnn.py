#!/usr/bin/env python
# coding:utf-8
"""
Author:
    LiTeng 1471356861@qq.com

Implement TextRNN, contains LSTM，GRU，RNN
Reference: "Effective LSTMs for Target-Dependent Sentiment Classification"
               "Bidirectional LSTM-CRF Models for Sequence Tagging"
               "Generative and discriminative text classification
                with recurrent neural networks"
"""

import tensorflow as tf
from tensorflow import keras
from util import Type


class RNNType(Type):
    RNN = 'RNN'
    LSTM = 'LSTM'
    GRU = 'GRU'

    @classmethod
    def str(cls):
        return ",".join([cls.RNN, cls.LSTM, cls.GRU])


class TextRNN(tf.keras.Model):
    """
    One layer rnn.
    """
    def __init__(self, config):
        super(TextRNN, self).__init__()
        self.config = config
        if  self.config.TextRNN.rnn_type == RNNType.LSTM:
            layer_cell = keras.layers.LSTM
        elif self.config.TextRNN.rnn_type == RNNType.GRU:
            layer_cell = keras.layers.GRU
        else:
            layer_cell = keras.layers.SimpleRNN

        self.rnn_type = config.TextRNN.rnn_type
        self.num_layers = config.TextRNN.num_layers
        self.bidirectional = config.TextRNN.bidirectional
        self.embedding = keras.layers.Embedding(config.TextRNN.input_dim, config.TextRNN.embedding_dimension,
                                                input_length=config.TextRNN.input_length)

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
                               kernel_regularizer=keras.regularizers.l2(self.config.TextRNN.l2 * 0.1),
                               recurrent_regularizer=keras.regularizers.l2(self.config.TextRNN.l2)))

        self.fc = keras.layers.Dense(config.TextRNN.num_classes)

    def call(self, inputs, training=None, mask=None):

        print('inputs', inputs)
        # [b, sentence len] => [b, sentence len, word embedding]
        x = self.embedding(inputs)
        print('embedding', x)
        for layer_cell in self.layer_cells:
            x = layer_cell(x)
        print('rnn', x)

        x = self.fc(x)
        print(x.shape)

        if self.config.logits_type == "softmax":
            x = tf.nn.softmax(x)
        elif self.config.logits_type == "sigmoid":
            x = tf.nn.sigmoid(x)

        return x


