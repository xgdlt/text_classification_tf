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


from util import Type


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