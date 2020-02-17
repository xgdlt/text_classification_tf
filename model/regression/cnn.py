#!usr/bin/env python
# coding:utf-8
# @time     :2019/10/30 10:51
# @author   :liteng
# @function :tensorflow2.0 for CNN
# @paper: Convolutional Neural Networks for Sentence Classiﬁcation


import tensorflow as tf
from tensorflow import keras


class model(tf.keras.Model):
    def __init__(self, config):
        super(model, self).__init__()
        self.config = config
        self.reshape = keras.layers.Reshape((config.CNN.input_length, config.CNN.input_dim, 1))

        self.conv = keras.layers.Conv2D(filters=64, kernel_size=(config.CNN.kernel_size, config.CNN.input_dim),
                                   strides=1, padding='valid', activation='relu')

        self.pool = tf.keras.layers.AveragePooling2D(pool_size=(config.CNN.input_length - 1 + 1, 1), padding='valid')

        #self.top_k = self.config.CNN.top_k_max_pooling
        self.flatten = keras.layers.Flatten()
        self.regression = keras.layers.Dense(1, name = "regression")
        self.classify = keras.layers.Dense(config.CNN.num_classes, name = "classify")

    def call(self, inputs,training=None, mask=None):
        print("inputs", inputs)
        if self.config.rest:
            x, base = inputs
        else:
            x = inputs

        x = self.reshape(x)
        print("reshape ", x)
        x = self.conv(x)
        x = self.pool(x)

        print("concat", x)
        x = self.flatten(x)
        print("flatten ", x)


        if self.config.rest:
            x = keras.layers.concatenate([x, base])
            print("concatenate ", x)

        regression = self.regression(x)

        if self.config.CNN.classify:
            classify = self.classify(x)
            classify = tf.nn.softmax(classify)
            return [regression, classify]
        return regression