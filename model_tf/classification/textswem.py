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
from util import Type


class SWEMType(Type):
    AVER = 'aver'
    MAX = 'max'
    CONCAT = 'concat'

    @classmethod
    def str(cls):
        return ",".join([cls.AVER, cls.MAX, cls.CONCAT])


class SWEM(tf.keras.Model):
    def __init__(self, config):
        super(SWEM, self).__init__()
        self.config = config

        self.embedding = keras.layers.Embedding(config.TextSWEM.input_dim, config.TextSWEM.embedding_dimension,
                                                input_length=config.TextSWEM.input_length)

        self.embedding_aver = keras.layers.GlobalAveragePooling1D()

        self.embedding_max = keras.layers.GlobalMaxPool1D()

        self.fc = keras.layers.Dense(config.TextSWEM.num_classes,activation='softmax')

    def call(self, inputs,training=True, mask=None):
        x = self.embedding(inputs)
        print("embedding", x)

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