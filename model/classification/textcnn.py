#!usr/bin/env python
# coding:utf-8
# @time     :2019/10/30 10:51
# @author   :liteng
# @function :tensorflow2.0 for textcnn
# @paper: Convolutional Neural Networks for Sentence Classiﬁcation


import tensorflow as tf
from tensorflow import keras


class TextCNN(tf.keras.Model):
    def __init__(self, config):
        super(TextCNN, self).__init__()
        self.config = config
        self.embedding = keras.layers.Embedding(config.TextCNN.input_dim, config.TextCNN.embedding_dimension,
                                                input_length=config.TextCNN.input_length)
        self.reshape = keras.layers.Reshape((config.TextCNN.input_length, config.TextCNN.embedding_dimension, 1))

        self.kernel_sizes = config.TextCNN.kernel_sizes
        self.convs = []
        self.pools = []

        for kernel_size in self.kernel_sizes:
            conv = keras.layers.Conv2D(filters=64, kernel_size=(kernel_size, config.TextCNN.embedding_dimension),
                                 strides=1, padding='valid', activation='relu')
            self.convs.append(conv)
            pool =  keras.layers.MaxPool2D(pool_size=(config.TextCNN.input_length - kernel_size + 1, 1), padding='valid')
            self.pools.append(pool)

        #self.top_k = self.config.TextCNN.top_k_max_pooling
        self.flatten = keras.layers.Flatten()
        self.fc = keras.layers.Dense(config.TextCNN.num_classes)

    def call(self, inputs,training=None, mask=None):
        print("inputs", inputs)
        x = self.embedding(inputs)
        print("embedding", x)
        x = self.reshape(x)
        print("reshape ", x)
        cnns = []
        for i in range(len(self.convs)):
            conv = self.convs[i](x)
            pool = self.pools[i](conv)
            cnns.append(pool)
            print("conv %d"%i, conv)
            print("pool %d"%i, pool)

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