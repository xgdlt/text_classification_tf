# coding=utf-8
#
# created by kpe on 20.Mar.2019 at 16:25
#

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf


class ActivationType(object):
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


def gelu(x):
    """
    Gelu activation from arXiv:1606.08415.
    See: https://arxiv.org/pdf/1606.08415.pdf
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        np.sqrt(2.0 / np.pi) * (x + 0.044715 * tf.pow(x, 3))
    ))
    return x * cdf


def gelu_exact(x):
    return x * tf.math.erfc(-x / tf.sqrt(2.)) / 2.


def get_activation(activation_string):
    if not isinstance(activation_string, str):
        return activation_string

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "sogmoid":
        return tf.nn.sigmoid
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "gelu_exact":
        return gelu_exact
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)
