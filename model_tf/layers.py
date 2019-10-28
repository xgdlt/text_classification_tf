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

import math

import tensorflow as tf
from tensorflow import keras

from model_tf.model_util import init_tensor
class k_max_pooling(keras.layers.Layer):
    """
        paper:        http://www.aclweb.org/anthology/P14-1062
        paper title:  A Convolutional Neural Network for Modelling Sentences
        Reference:    https://stackoverflow.com/questions/51299181/how-to-implement-k-max-pooling-in-tensorflow-or-keras
        动态K-max pooling
            k的选择为 k = max(k, s * (L-1) / L)
            其中k为预先选定的设置的最大的K个值，s为文本最大长度，L为第几个卷积层的深度（单个卷积到连接层等）
        github tf实现可以参考: https://github.com/lpty/classifier/blob/master/a04_dcnn/model.py
    """
    def __init__(self, top_k=8, **kwargs):
        self.top_k = top_k
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        inputs_reshape = tf.transpose(inputs, perm=[0, 2, 1])
        pool_top_k = tf.nn.top_k(input=inputs_reshape, k=self.top_k, sorted=False).values
        pool_top_k_reshape = tf.transpose(pool_top_k, perm=[0, 2, 1])
        return pool_top_k_reshape

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.top_k, input_shape[-1]





class BasicBlock(keras.layers.Layer):

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = keras.layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation('relu')

        self.conv2 = keras.layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = keras.layers.BatchNormalization()

        if stride != 1:
            self.downsample = keras.Sequential()
            self.downsample.add(keras.layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x:x


    def call(self, inputs, training=None):

        # [b, h, w, c]
        out = self.conv1(inputs)
        out = self.bn1(out,training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out,training=training)

        identity = self.downsample(inputs)

        output = keras.layers.add([out, identity])
        output = tf.nn.relu(output)

        return output
