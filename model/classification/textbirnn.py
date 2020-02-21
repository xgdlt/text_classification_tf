#!/usr/bin/env python
# coding:utf-8
"""
Author:
    LiTeng 1471356861@qq.com

Implement TextBiRNN, contains LSTM，GRU，RNN
Reference: "Effective LSTMs for Target-Dependent Sentiment Classification"
               "Bidirectional LSTM-CRF Models for Sequence Tagging"
               "Generative and discriminative text classification
                with recurrent neural networks"
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
            self.reshape = keras.layers.Reshape((config.TextBiRNN.input_length, config.TextBiRNN.embedding_dimension))
            
        if  self.config.TextBiRNN.rnn_type == RNNType.LSTM:
            layer_cell = keras.layers.LSTM
        elif self.config.TextBiRNN.rnn_type == RNNType.GRU:
            layer_cell = keras.layers.GRU
        else:
            layer_cell = keras.layers.SimpleRNN

        self.rnn_type = config.TextBiRNN.rnn_type
        self.num_layers = config.TextBiRNN.num_layers
        self.bidirectional = config.TextBiRNN.bidirectional
        #self.embedding = keras.layers.Embedding(config.TextBiRNN.input_dim, config.TextBiRNN.embedding_dimension,
        #                                        input_length=config.TextBiRNN.input_length)

        self.layer_cells = []
        for i in range(config.TextBiRNN.num_layers):
            self.layer_cells.append(keras.layers.Bidirectional(
                layer_cell(config.TextBiRNN.hidden_dimension,
                            use_bias=config.TextBiRNN.use_bias,
                            activation=config.TextBiRNN.activation,
                            kernel_regularizer=keras.regularizers.l2(self.config.TextBiRNN.l2 * 0.1),
                            recurrent_regularizer=keras.regularizers.l2(self.config.TextBiRNN.l2))))

        self.fc = keras.layers.Dense(config.TextBiRNN.num_classes)

    def call(self, inputs, training=None, mask=None):

        print("inputs", inputs)
        x = inputs
        if self.config.embedding.use_embedding:
            # [b, sentence len] => [b, sentence len, word embedding]
            x = self.embedding(x)
            print("embedding", x)
        else:
            x = self.reshape(x)
       
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


