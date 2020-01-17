#!usr/bin/env python
# coding:utf-8
# @time     :2019/10/30 10:51
# @author   :liteng
# @function :tensorflow2.0 for CNN
# @paper: Convolutional Neural Networks for Sentence Classiﬁcation


import tensorflow as tf
from tensorflow import keras


class CNN(tf.keras.Model):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        self.reshape = keras.layers.Reshape((config.CNN.input_length, config.CNN.input_dim, 1))

        self.conv = keras.layers.Conv2D(filters=64, kernel_size=(config.CNN.kernel_size, config.CNN.input_dim),
                                   strides=1, padding='valid', activation='relu')

        self.pool = keras.layers.MaxPool2D(pool_size=(config.CNN.input_length - 1 + 1, 1), padding='valid')


        #self.top_k = self.config.CNN.top_k_max_pooling
        self.flatten = keras.layers.Flatten()
        self.fc = keras.layers.Dense(config.CNN.num_classes)

    def call(self, inputs,training=None, mask=None):
        print("inputs", inputs)
        x = self.embedding(inputs)
        print("embedding", x)
        x = self.reshape(x)
        print("reshape ", x)
        x = self.conv(x)
        x = self.pool(x)

        print("concat", x)
        x = self.flatten(x)
        print("flatten ", x)
        x = self.fc(x)
        if self.config.logits_type == "softmax":
            x = tf.nn.softmax(x)
        elif self.config.logits_type == "sigmoid":
            x = tf.nn.sigmoid(x)
        return x