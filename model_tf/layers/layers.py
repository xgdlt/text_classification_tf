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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs,top_k=None):
        if not top_k:
            top_k = inputs.shape[1] / 2
        inputs_reshape = tf.transpose(inputs, perm=[0, 2, 1])
        pool_top_k = tf.nn.top_k(input=inputs_reshape, k=top_k, sorted=False).values
        pool_top_k_reshape = tf.transpose(pool_top_k, perm=[0, 2, 1])
        return pool_top_k_reshape

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.top_k, input_shape[-1]



class dynamic_k_max_pooling(keras.layers.Layer):
    """
        paper:        http://www.aclweb.org/anthology/P14-1062
        paper title:  A Convolutional Neural Network for Modelling Sentences
        Reference:    https://stackoverflow.com/questions/51299181/how-to-implement-k-max-pooling-in-tensorflow-or-keras
        动态K-max pooling
            k的选择为 k = max(k, s * (L-1) / L)
            其中k为预先选定的设置的最大的K个值，s为文本最大长度，L为第几个卷积层的深度（单个卷积到连接层等）
        github tf实现可以参考: https://github.com/lpty/classifier/blob/master/a04_dcnn/model.py
    """
    def __init__(self, top_k=3, **kwargs):
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


class prem_fold(keras.layers.Layer):
    """
        paper:       http://www.aclweb.org/anthology/P14-1062
        paper title: A Convolutional Neural Network for Modelling Sentences
        detail:      垂直于句子长度的方向，相邻值相加，就是embedding层300那里，（0+1,2+3...298+299）
        github tf实现可以参考: https://github.com/lpty/classifier/blob/master/a04_dcnn/model.py
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, conv_shape):
        super().build(conv_shape)

    def call(self, convs):
        conv1 = convs[:, :, ::2]
        conv2 = convs[:, :, 1::2]
        conv_fold = keras.layers.add([conv1, conv2])
        return conv_fold

    def compute_output_shape(self, conv_shape):
        return conv_shape[0], conv_shape[1], int(conv_shape[2] / 2)


class wide_convolution(keras.layers.Layer):
    """
        paper: http://www.aclweb.org/anthology/P14-1062
        paper title: "A Convolutional Neural Network for Modelling Sentences"
        宽卷积, 如果s表示句子最大长度, m为卷积核尺寸,
           则宽卷积输出为 s + m − 1,
           普通卷积输出为 s - m + 1.
        github keras实现可以参考: https://github.com/AlexYangLi/TextClassification/blob/master/models/keras_dcnn_model.py
    """
    def __init__(self, filters=64, kernel_size=3, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = keras.layers.ZeroPadding1D((self.filters - 1, self.filters - 1))
        self.conv = keras.layers.Conv1D(filters=self.filters,
                            kernel_size=self.kernel_size,
                            strides=1,
                            padding='VALID',
                            kernel_initializer='normal',  # )(x_input_pad)
                            activation='tanh')
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        x_input_pad =  self.padding(inputs)
        conv_1d = self.conv (x_input_pad)
        return conv_1d

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] + self.filters - 1, input_shape[-1]


class BasicBlock(keras.layers.Layer):


    def __init__(self, filter_num=64, stride=1,shortcut=True, pool_type= None):
        super(BasicBlock, self).__init__()
        self.shortcut = shortcut
        self.pool_type = pool_type
        self.conv1 = keras.layers.Conv1D(filters=filter_num, kernel_size=3, strides=stride, padding='same')
        self.bn1 = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation('relu')

        self.conv2 = keras.layers.Conv1D(filters=filter_num, kernel_size=3, strides=stride, padding='same')
        self.bn2 = keras.layers.BatchNormalization()

        if shortcut:
            self.shortcut_conv =  keras.layers.Conv1D(filters=filter_num, kernel_size=1, strides=2, padding='same')
            self.shortcut_bn =  keras.layers.BatchNormalization()

        if pool_type == 'max':
            self.downsampl = keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='SAME')
        elif pool_type == 'k-max':
            self.downsampl = k_max_pooling()
        elif pool_type == 'conv':
            self.downsampl = keras.layers.Conv1D(filters=filter_num,kernel_size=3, strides=2,padding='SAME')
        else:
            self.downsampl = keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='SAME')

        #self.downsampl = self.downsampling(pool_type)

        if pool_type is not None: # filters翻倍
            self.pool_conv = keras.layers.Conv1D(filters=filter_num*2, kernel_size=1, strides=1, padding='SAME')
            self.pool_bn = keras.layers.BatchNormalization()


    def call(self, inputs, training=None):

        # [b, h, w, c]
        out = self.conv1(inputs)
        out = self.bn1(out,training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out,training=training)

        if self.shortcut:
            conv = self.shortcut_conv(inputs)
            bn_out = self.shortcut_bn(conv)
            output = self.downsampl(out)
            out = keras.layers.add([output, bn_out])
        else:
            out =  self.relu(out)
            out = self.downsampl(out)

        if self.pool_type is not None:  # filters翻倍
            out = self.pool_conv(out)
            out = self.pool_bn(out)

        return out


class ResCNN(keras.layers.Layer):

    def __init__(self, filters=64, kernel_size=1, stride=1, shortcut=True, pool_type=None):
        super(ResCNN, self).__init__()
        self.shortcut = shortcut
        self.pool_type = pool_type
        self.conv1 = keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, padding='same')
        self.bn1 = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation('relu')

        self.conv2 = keras.layers.Conv1D(filters=filters, kernel_size=3, strides=stride, padding='same')
        self.bn2 = keras.layers.BatchNormalization()


    def call(self, inputs, training=None):

        # [b, h, w, c]
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        #out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.relu(out)


        return out

    def downsampling(self,filter_num=64, pool_type='conv'):
        """
            In addition, downsampling with stride 2 essentially doubles the effective coverage
            (i.e., coverage in the original document) of the convolution kernel;
            therefore, after going through downsampling L times,
            associations among words within a distance in the order of 2L can be represented.
            Thus, deep pyramid CNN is computationally efﬁcient for representing long-range associations
            and so more global information.
            参考: https://github.com/zonetrooper32/VDCNN/blob/keras_version/vdcnn.py
        :param inputs: tensor,
        :param pool_type: str, select 'max', 'k-max' or 'conv'
        :return: tensor,
        """
        if pool_type == 'max':
            pooling = keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='SAME')
        elif pool_type == 'k-max':
            pooling = k_max_pooling()
        elif pool_type == 'conv':
            pooling = keras.layers.Conv1D(filters=filter_num,kernel_size=3, strides=2, padding='SAME')

        return pooling
