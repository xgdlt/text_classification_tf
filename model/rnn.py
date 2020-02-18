#!/usr/bin/env python
# coding:utf-8
"""
Author:
    LiTeng 1471356861@qq.com
"""

import tensorflow as tf
from tensorflow import keras

from utils.util import Type


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
    def __init__(self, units,input_dim,output_dim,input_length, activation='tanh',
                 use_bias=True, dropout=0.0, num_layers=1,
                 bidirectional=False, rnn_type=RNNType.GRU):
        super(RNN, self).__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding = keras.layers.Embedding(input_dim, output_dim, input_length=input_length)

        if rnn_type == RNNType.LSTM:
            if bidirectional:
                self.rnn = tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(units,use_bias=use_bias,activation = activation))
            else:
                self.rnn = tf.keras.layers.LSTM(units,use_bias=use_bias,activation = activation)
        elif rnn_type == RNNType.GRU:
            if bidirectional:
                self.rnn = tf.keras.layers.Bidirectional(
                    tf.keras.layers.GRU(units, use_bias=use_bias, activation=activation))
            else:
                self.rnn = tf.keras.layers.GRU(units,use_bias=use_bias,activation = activation)
        elif rnn_type == RNNType.RNN:
            if bidirectional:
                self.rnn = tf.keras.layers.Bidirectional(
                     tf.keras.layers.SimpleRNN(units, use_bias=use_bias, activation=activation))
            else:
                self.rnn = tf.keras.layers.SimpleRNN(units,use_bias=use_bias,activation = activation)
        else:
            raise TypeError(
                "Unsupported rnn init type: %s. Supported rnn type is: %s" % (
                    rnn_type, RNNType.str()))
        self.fc = keras.layers.Dense(1)

    def call(self, inputs, training=None, mask=None):

        # print('x', inputs.shape)
        # [b, sentence len] => [b, sentence len, word embedding]
        x = self.embedding(inputs)
        # print('embedding', x.shape)
        x = self.rnn(x)
        # print('rnn', x.shape)

        x = self.fc(x)
        print(x.shape)

        return x


def main():

    units = 64
    num_classes = 2
    batch_size = 32
    epochs = 20

    model = RNN(units, num_classes, num_layers=2)


    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # train
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_data=(x_test, y_test), verbose=1)

    # evaluate on test set
    scores = model.evaluate(x_test, y_test, batch_size, verbose=1)
    print("Final test loss and accuracy :", scores)

