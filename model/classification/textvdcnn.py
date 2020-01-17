#!/usr/bin/env python
# coding:utf-8
"""
Author:
    LiTeng 1471356861@qq.com

Implement model of "Very deep convolutional networks for text classification"
which can be seen at "http://www.aclweb.org/anthology/E17-1104"
"""

import tensorflow as tf
from tensorflow import keras

from model.layers.layers import k_max_pooling


class ConvBlock(keras.layers.Layer):


    def __init__(self, filter_num=64, stride=1,shortcut=True, pool_type= None):
        super(ConvBlock, self).__init__()
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
            bn_out = self.shortcut_bn(conv,training=training)
            output = self.downsampl(out)
            out = keras.layers.add([output, bn_out])
        else:
            out =  self.relu(out)
            out = self.downsampl(out)

        if self.pool_type is not None:  # filters翻倍
            out = self.pool_conv(out)
            out = self.pool_bn(out,training=training)

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



class IdentityBlock(keras.layers.Layer):


    def __init__(self, filter_num=64,kernel_size=3, stride=1,shortcut=True):
        super(IdentityBlock, self).__init__()
        self.shortcut = shortcut
        self.conv1 = keras.layers.Conv1D(filters=filter_num, kernel_size=kernel_size, strides=stride, padding='same')
        self.bn1 = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation('relu')

        self.conv2 = keras.layers.Conv1D(filters=filter_num, kernel_size=kernel_size, strides=stride, padding='same')
        self.bn2 = keras.layers.BatchNormalization()

    def call(self, inputs, training=None):

        # [b, h, w, c]
        out = self.conv1(inputs)
        out = self.bn1(out,training=training)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out,training=training)

        if self.shortcut:
            out = keras.layers.add([out, inputs])
            out = self.relu(out)
        return out

class TextVDCNN(keras.Model):
    def __init__(self,  config):
        """all convolutional blocks
        4 kinds of conv blocks, which #feature_map are 64,128,256,512
        Depth:             9  17 29 49
        ------------------------------
        conv block 512:    2  4  4  6
        conv block 256:    2  4  4  10
        conv block 128:    2  4  10 16
        conv block 64:     2  4  10 16
        First conv. layer: 1  1  1  1
        """
        super(TextVDCNN, self).__init__()

        self.vdcnn_num_convs = {}
        self.vdcnn_num_convs[9] = [2, 2, 2, 2]
        self.vdcnn_num_convs[17] = [4, 4, 4, 4]
        self.vdcnn_num_convs[29] = [10, 10, 4, 4]
        self.vdcnn_num_convs[49] = [16, 16, 10, 6]
        self.num_kernels = [64, 128, 256, 512]
        self.config = config
        self.vdcnn_depth = config.TextVDCNN.vdcnn_depth
        self.embedding = keras.layers.Embedding(config.TextVDCNN.input_dim, config.TextVDCNN.embedding_dimension,
                                                input_length=config.TextVDCNN.input_length)

        self.first_conv = keras.layers.Conv1D(filters=64, kernel_size=3,
                            strides=1, padding='SAME', activation='relu')


        last_num_kernel = 64
        self.identity_blocks = []
        self.con_blocks = []
        for i, num_kernel in enumerate(self.num_kernels):
            tmp_identity_blocks = []
            for j in range(0, self.vdcnn_num_convs[self.vdcnn_depth][i]-1):
                tmp_identity_blocks.append(IdentityBlock(filter_num=num_kernel))
            self.identity_blocks.append(tmp_identity_blocks)
            self.con_blocks.append(ConvBlock(filter_num=num_kernel, pool_type=self.config.TextVDCNN.pool_type))

        self.relu = tf.keras.layers.ReLU()

        self.top_k = self.config.TextVDCNN.top_k_max_pooling
        self.k_max_pooling = k_max_pooling()
        self.flatten = keras.layers.Flatten()
        self.fc1 = keras.layers.Dense(2048, activation="relu")
        self.fc2 = keras.layers.Dense(2048, activation="relu")
        self.fc = keras.layers.Dense(config.TextVDCNN.num_classes)



    def call(self, inputs, training=None, mask=None,logits_type=None):

        x = self.embedding(inputs)
        print("embedding", x)
        # first conv layer (kernel_size=3, #feature_map=64)
        out = self.first_conv(x)
        print("first_conv", out)
        out = self.relu (out)
        print("first_conv_relu", out)

        # all convolutional blocks

        for i in range(0, len(self.num_kernels)):
            for identity_block in self.identity_blocks[i]:
                out = identity_block(out)
                print("identity_block_%d", i, out)
            out = self.con_blocks[i](out)
            print("con_blocks_%d", i, out)
        out = self.k_max_pooling(out, self.top_k)
        print("k_max_pooling", out)

        out = self.flatten(out)
        print("flatten", out)

        out = self.fc1(out)
        print("fc1", out)

        out = self.fc2(out)
        print("fc2", out)

        out = self.fc(out)
        print("fc", out)

        if logits_type == "softmax":
            out = tf.nn.softmax(out)
        elif logits_type == "sigmoid":
            out = tf.nn.sigmoid(out)
        return out
