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

import codecs as cs

import numpy as np
import tensorflow as tf

from utils.util import Type


class ActivationType(Type):
    """Standard names for activation
    """
    SIGMOID = 'sigmoid'
    TANH = "tanh"
    RELU = 'relu'
    LEAKY_RELU = 'leaky_relu'
    NONE = 'linear'

    @classmethod
    def str(cls):
        return ",".join(
            [cls.SIGMOID, cls.TANH, cls.RELU, cls.LEAKY_RELU, cls.NONE])


class InitType(Type):
    """Standard names for init
    """
    UNIFORM = 'uniform'
    NORMAL = "normal"
    XAVIER_UNIFORM = 'xavier_uniform'
    XAVIER_NORMAL = 'xavier_normal'
    KAIMING_UNIFORM = 'kaiming_uniform'
    KAIMING_NORMAL = 'kaiming_normal'
    ORTHOGONAL = 'orthogonal'

    def str(self):
        return ",".join(
            [self.UNIFORM, self.NORMAL, self.XAVIER_UNIFORM, self.XAVIER_NORMAL,
             self.KAIMING_UNIFORM, self.KAIMING_NORMAL, self.ORTHOGONAL])


class FAN_MODE(Type):
    """Standard names for fan mode
    """
    FAN_IN = 'FAN_IN'
    FAN_OUT = "FAN_OUT"

    def str(self):
        return ",".join([self.FAN_IN, self.FAN_OUT])

class OptimizerType(Type):
    """Standard names for optimizer
    """
    ADAM = "Adam"
    ADADELTA = "Adadelta"
    BERT_ADAM = "BERTAdam"

    def str(self):
        return ",".join([self.ADAM, self.ADADELTA])

def get_hierar_relations(hierar_taxonomy, label_map):
    """ get parent-children relationships from given hierar_taxonomy
        hierar_taxonomy: parent_label \t child_label_0 \t child_label_1 \n
    """
    hierar_relations = {}
    with cs.open(hierar_taxonomy, "r", "utf8") as f:
        for line in f:
            line_split = line.strip("\n").split("\t")
            parent_label, children_label = line_split[0], line_split[1:]
            if parent_label not in label_map:
                continue
            parent_label_id = label_map[parent_label]
            children_label_ids = [label_map[child_label] \
                for child_label in children_label if child_label in label_map]
            hierar_relations[parent_label_id] = children_label_ids
    return hierar_relations



def select_k(len_max, length_conv, length_curr, k_con=3):
    """
        dynamic k max pooling中的k获取
    :param len_max:int, max length of input sentence
    :param length_conv: int, deepth of all convolution layer
    :param length_curr: int, deepth of current convolution layer
    :param k_con: int, k of constant
    :return: int, return
    """
    if length_conv >= length_curr:
        k_ml = int(len_max * (length_conv-length_curr) / length_conv)
        k = max(k_ml, k_con)
    else:
        k = k_con
    return k


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
    # 编码器填充遮挡
    enc_padding_mask = create_padding_mask(inp)

    # 在解码器的第二个注意力模块使用。
    # 该填充遮挡用于遮挡编码器的输出。
    dec_padding_mask = create_padding_mask(inp)

    # 在解码器的第一个注意力模块使用。
    # 用于填充（pad）和遮挡（mask）解码器获取到的输入的后续标记（future tokens）。
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # 添加额外的维度来将填充加到
    # 注意力对数（logits）。
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)