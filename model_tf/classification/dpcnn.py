#!usr/bin/env python
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


class ResCNN(keras.layers.Layer):

    def __init__(self, filters=64, kernel_size=3, stride=1):
        super(ResCNN, self).__init__()
        self.conv1 = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, padding='same')
        self.bn1 = keras.layers.BatchNormalization()
        self.relu =  keras.layers.PReLU()

        self.conv2 = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, padding='same')
        self.bn2 = keras.layers.BatchNormalization()

    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        print("conv1", out)
        out = self.bn1(out, training=training)
        #out = self.relu(out)

        out = self.conv2(out)
        print("conv2", out)
        out = self.bn2(out, training=training)
        out = self.relu(out)

        return out


class Repeat(keras.layers.Layer):

    def __init__(self, filters=64, kernel_size=3, stride=1, globa=False):
        super(Repeat, self).__init__()
        self.res = ResCNN(filters=filters, kernel_size=kernel_size, stride=stride)
        self.add = keras.layers.Add()
        if globa:
            self.pool = keras.layers.GlobalMaxPool1D()
        else:
            self.pool = keras.layers.MaxPool1D(pool_size=3, strides=2)

    def call(self, inputs, training=None):

        out = self.res(inputs, training=training)
        print("Repeat_rescnn", out)
        out = self.add([out, inputs])
        out = self.pool(out)
        print("Repeat pool", out)
        return out

class DPCNN(keras.Model):
    """
    Reference:
        Deep Pyramid Convolutional Neural Networks for Text Categorization
    """

    def __init__(self,  config):
        super(DPCNN, self).__init__(config)
        self.config = config
        self.embedding = keras.layers.Embedding(config.DPCNN.input_dim, config.DPCNN.embedding_dimension,
                                                input_length=config.DPCNN.input_length)
        self.spatial_dropout1d = keras.layers.SpatialDropout1D(config.DPCNN.spatial_dropout)
        self.conv_1 = keras.layers.Conv1D(filters=config.DPCNN.filters, kernel_size=1, padding='SAME', \
                                          kernel_regularizer=keras.regularizers.l2(config.DPCNN.l2))
        self.prelu = keras.layers.PReLU()

        self.first_block = ResCNN(filters=config.DPCNN.filters)
        self.add = keras.layers.Add()
        self.max_pooling = keras.layers.MaxPool1D(pool_size=3, strides=2)

        self.repeats = []
        for i in range(0,config.DPCNN.repeat-1):
            self.repeats.append(Repeat(filters=config.DPCNN.filters))
        self.repeats.append(Repeat(filters=config.DPCNN.filters, globa=True))

        self.fc1 = keras.layers.Dense(256)
        self.bn1 = keras.layers.BatchNormalization()
        self.fc_prelu = keras.layers.PReLU()

        self.dropout = keras.layers.Dropout(config.DPCNN.spatial_dropout)
        self.fc = keras.layers.Dense(config.DPCNN.num_classes)


    def call(self, inputs,training=None, mask=None):
        embedding = self.embedding(inputs)
        embedding = self.spatial_dropout1d(embedding)

        print("embedding",embedding )

        region_embedding = self.conv_1(embedding)
        region_embedding = self.prelu(region_embedding)
        print("region_embedding", region_embedding)

        out = self.first_block(embedding,training= training)
        out = self.add([out,region_embedding])
        out = self.max_pooling(out)
        print("max_pooling", out)

        for i in range(0, self.config.DPCNN.repeat):
            out =  self.repeats[i](out, training= training)

        out = self.fc1 (out)
        print("fc1", out)
        out = self.bn1(out)
        print("bn1", out)
        out = self.fc_prelu(out)
        print("prelu", out)
        out = self.dropout(out)
        out = self.fc(out)
        print("fc", out)

        if self.config.logits_type == "softmax":
            out = tf.nn.softmax(out)
        elif self.config.logits_type == "sigmoid":
            out = tf.nn.sigmoid(out)

        return  out