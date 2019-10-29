#!/usr/bin/env python
# coding:utf-8
"""
Tencent is pleased to support the open source community by making NeuralClassifier available.
Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the MIT License (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at
http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing, software distributed under the License
is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied. See the License for thespecific language governing permissions and limitations under
the License.
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


class RNN(tf.keras.Model):
    """
    One layer rnn.
    """
    def __init__(self, config):
        super(RNN, self).__init__()
        self.rnn_type = config.TextRNN.rnn_type
        self.num_layers = config.TextRNN.num_layers
        self.bidirectional = config.TextRNN.bidirectional
        self.embedding = keras.layers.Embedding(config.TextRNN.input_dim, config.TextRNN.embedding_dimension,
                                                input_length=config.TextRNN.input_length)

        if self.rnn_type == RNNType.LSTM:
            if self.bidirectional:
                self.rnn = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(config.TextRNN.hidden_dimension,
                                         use_bias=config.TextRNN.use_bias,
                                         activation=config.TextRNN.activation))
            else:
                self.rnn = tf.keras.layers.LSTM(config.TextRNN.hidden_dimension,
                                                use_bias=config.TextRNN.use_bias,
                                                activation=config.TextRNN.activation)
        elif self.rnn_type == RNNType.GRU:
            if self.bidirectional:
                self.rnn = tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(config.TextRNN.hidden_dimension,
                                        use_bias=config.TextRNN.use_bias,
                                        activation=config.TextRNN.activation))
            else:
                self.rnn = tf.keras.layers.GRU(config.TextRNN.hidden_dimension,
                                               use_bias=config.TextRNN.use_bias,
                                               activation=config.TextRNN.activation)
        elif self.rnn_type == RNNType.RNN:
            if self.bidirectional:
                self.rnn = tf.keras.layers.Bidirectional(
                     tf.keras.layers.SimpleRNN(config.TextRNN.hidden_dimension,
                                               use_bias=config.TextRNN.use_bias,
                                               activation=config.TextRNN.activation))
            else:
                self.rnn = tf.keras.layers.SimpleRNN(config.TextRNN.hidden_dimension,
                                                     use_bias=config.TextRNN.use_bias,
                                                     activation=config.TextRNN.activation)
        else:
            raise TypeError(
                "Unsupported rnn init type: %s. Supported rnn type is: %s" % (
                    config.TextRNN.rnn_type, RNNType.str()))
        self.fc = keras.layers.Dense(config.TextCNN.num_classes,activation='softmax')

    def call(self, inputs, training=None, mask=None):

        print('inputs', inputs)
        # [b, sentence len] => [b, sentence len, word embedding]
        x = self.embedding(inputs)
        print('embedding', x)
        x = self.rnn(x)
        print('rnn', x)

        x = self.fc(x)
        print(x.shape)

        return x


