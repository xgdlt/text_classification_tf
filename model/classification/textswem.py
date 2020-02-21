#!usr/bin/env python
# coding:utf-8

"""
Author:
    LiTeng 1471356861@qq.com
"""
import tensorflow as tf
from tensorflow import keras

from model.layers.embeddings import EmbeddingsLayer
from utils.logger import Type


class SWEMType(Type):
    AVER = 'aver'
    MAX = 'max'
    CONCAT = 'concat'

    @classmethod
    def str(cls):
        return ",".join([cls.AVER, cls.MAX, cls.CONCAT])


class Model(tf.keras.Model):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        if self.config.embedding.use_embedding:
            self.embedding = EmbeddingsLayer(config.embedding)
        else:
            self.reshape = keras.layers.Reshape((config.TextSWEM.input_length, config.TextSWEM.embedding_dimension))

        self.embedding_aver = keras.layers.GlobalAveragePooling1D()

        self.embedding_max = keras.layers.GlobalMaxPool1D()

        self.fc = keras.layers.Dense(config.TextSWEM.num_classes,activation='softmax')

    def call(self, inputs,training=True, mask=None):
        print("inputs", inputs)
        x = inputs
        if self.config.embedding.use_embedding:
            # [b, sentence len] => [b, sentence len, word embedding]
            x = self.embedding(x)
            print("embedding", x)
        else:
            x = self.reshape(x)

        if self.config.TextSWEM.type == SWEMType.AVER:
            x = self.embedding_aver(x)

        elif self.config.TextSWEM.type == SWEMType.MAX:
            x = self.embedding_max(x)
        else:
            x_aver = self.embedding_aver(x)
            print("embedding_aver", x)
            x_max = self.embedding_max(x)
            print("embedding_max", x)
            x = keras.layers.concatenate([x_aver,x_max])

        print("embedding_encode", x)

        #x = self.flatten(x)
        #print("flatten ", x)
        x = self.fc(x)
        return x