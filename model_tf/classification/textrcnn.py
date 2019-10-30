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


class TextRCNN(tf.keras.Model):
    """
    One layer rnn.
    """
    def __init__(self, config):
        super(TextRCNN, self).__init__()

        self.embedding = keras.layers.Embedding(config.TextRNN.input_dim, config.TextRNN.embedding_dimension,
                                                input_length=config.TextRNN.input_length)
        self.config = config
        if config.TextRNN.rnn_type == RNNType.LSTM:
            if config.TextRNN.bidirectional:
                self.rnn = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(config.TextRNN.hidden_dimension,
                                         use_bias=config.TextRNN.use_bias,
                                         activation=config.TextRNN.activation))
            else:
                self.rnn = tf.keras.layers.LSTM(config.TextRNN.hidden_dimension,
                                                use_bias=config.TextRNN.use_bias,
                                                activation=config.TextRNN.activation)
        elif config.TextRNN.rnn_type == RNNType.GRU:
            if config.TextRNN.bidirectional:
                self.rnn = tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(config.TextRNN.hidden_dimension,
                                        use_bias=config.TextRNN.use_bias,
                                        activation=config.TextRNN.activation))
            else:
                self.rnn = tf.keras.layers.GRU(config.TextRNN.hidden_dimension,
                                               use_bias=config.TextRNN.use_bias,
                                               activation=config.TextRNN.activation)
        elif config.TextRNN.rnn_type == RNNType.RNN:
            if config.TextRNN.bidirectional:
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


    def call(self, inputs, training=None, mask=None):

        print('inputs', inputs)
        # [b, sentence len] => [b, sentence len, word embedding]
        x = self.embedding(inputs)
        print('embedding', x)
        x = self.rnn(x)
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
        return x


