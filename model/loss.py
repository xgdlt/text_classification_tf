#!/usr/bin/env python
# coding:utf-8
"""
Author:
    LiTeng 1471356861@qq.com
"""


from utils.logger import Type
from tensorflow import keras
import tensorflow as tf

class LossType(Type):
    """Standard names for loss type
    """
    SOFTMAX_CROSS_ENTROPY = "SoftmaxCrossEntropy"
    SPARSE_SOFTMAX_CROSS_ENTROPY = "SparseSoftmaxCrossEntropy"
    SOFTMAX_FOCAL_CROSS_ENTROPY = "SoftmaxFocalCrossEntropy"
    SIGMOID_FOCAL_CROSS_ENTROPY = "SigmoidFocalCrossEntropy"
    MEAN_SQUARED_ERROR = "MeanSquaredError"

    @classmethod
    def str(cls):
        return ",".join([cls.SOFTMAX_CROSS_ENTROPY,
                         cls.SOFTMAX_FOCAL_CROSS_ENTROPY,
                         cls.SIGMOID_FOCAL_CROSS_ENTROPY,
                         cls.MEAN_SQUARED_ERROR])


class ActivationType(Type):
    """Standard names for activation type
    """
    SOFTMAX = "Softmax"
    SIGMOID = "Sigmoid"

    @classmethod
    def str(cls):
        return ",".join([cls.SOFTMAX,
                         cls.SIGMOID])


class FocalLoss(tf.keras.losses.Loss):
    """Softmax focal loss
    references: Focal Loss for Dense Object Detection
                https://github.com/Hsuxu/FocalLoss-PyTorch
    """

    def __init__(self, label_size, activation_type=ActivationType.SOFTMAX,from_logits=False,
                 gamma=2.0, alpha=0.25, epsilon=1.e-9):
        super(FocalLoss, self).__init__()
        self.num_cls = label_size
        self.activation_type = activation_type
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == ActivationType.SOFTMAX:
            if self.from_logits:
                y_pred = tf.softmax(y_pred)
            y_true = tf.cast(y_true, tf.float32)
            alpha_t = y_true *  self.alpha + (tf.ones_like(y_true) - y_true) * (1 -  self.alpha)

            p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred) + self.epsilon
            focal_loss = - alpha_t * tf.pow((tf.ones_like(y_true) - p_t), self.gamma) * tf.log(p_t)

            loss = tf.reduce_mean(focal_loss)
        elif self.activation_type == ActivationType.SIGMOID:
            if self.from_logits:
                y_pred = tf.sigmoid(y_pred)
            y_true = tf.cast(y_true, tf.float32)
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.clip_by_value(y_pred, self.epsilon, 1. - self.epsilon)

            alpha_t = y_true * self.alpha + (tf.ones_like(y_true) - y_true) * (1 - self.alpha)
            y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
            ce = -tf.log(y_t)
            weight = tf.pow(tf.subtract(1., y_t), self.gamma)
            fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
            loss = tf.reduce_mean(fl)


        else:
            raise TypeError("Unknown activation type: " + self.activation_type
                            + "Supported activation types: " +
                            ActivationType.str())
        return loss


def get_classify_loss(type,from_logits=False,num_classes=None,gamma=2.0, alpha=0.25, epsilon=1.e-9):
    if type == LossType.SOFTMAX_CROSS_ENTROPY:
        criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=from_logits)
    elif type == LossType.SPARSE_SOFTMAX_CROSS_ENTROPY:
        criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits)
    elif type == LossType.SOFTMAX_FOCAL_CROSS_ENTROPY:
        criterion = FocalLoss(num_classes, ActivationType.SOFTMAX, from_logits, gamma, alpha)
    elif type == LossType.SIGMOID_FOCAL_CROSS_ENTROPY:
        criterion = FocalLoss(num_classes, ActivationType.SIGMOID, from_logits,   gamma, alpha, epsilon)
    elif type == LossType.MEAN_SQUARED_ERROR:
        criterion = keras.losses.MeanSquaredError()
    else:
        raise TypeError(
            "Unsupported loss type: %s. Supported loss type is: %s" % (
                type, LossType.str()))
    return criterion


def get_loss(config):
    if config.loss.type == LossType.SOFTMAX_CROSS_ENTROPY:
        criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=config.loss.from_logits)
    elif config.loss.type == LossType.SPARSE_SOFTMAX_CROSS_ENTROPY:
        criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=config.loss.from_logits)
    elif config.loss.type == LossType.SOFTMAX_FOCAL_CROSS_ENTROPY:
        criterion = FocalLoss(config.num_classes, ActivationType.SOFTMAX, config.loss.from_logits,
                              config.loss.focal_loss.gamma, config.loss.focal_loss.alpha)
    elif config.loss.type == LossType.SIGMOID_FOCAL_CROSS_ENTROPY:
        criterion = FocalLoss(config.num_classes, ActivationType.SIGMOID, config.loss.from_logits,
                              config.loss.focal_loss.gamma, config.loss.focal_loss.alpha, config.loss.focal_loss.epsilon)
    elif config.loss.type == LossType.MEAN_SQUARED_ERROR:
        criterion = keras.losses.MeanSquaredError()
    else:
        raise TypeError(
            "Unsupported loss type: %s. Supported loss type is: %s" % (
                config.loss.type, LossType.str()))
    return criterion

