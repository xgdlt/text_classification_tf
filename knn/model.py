# -*- coding: utf-8 -*-
#TextCNN: 1. embeddding layers, 2.convolutional layer, 3.max-pooling, 4.softmax layer.
# print("started...")
import tensorflow as tf

class KNN:
    def __init__(self, sequence_length, batch_size=64):

        #self.trains = trains
        #self.labels = labels
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.train_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="train_x")  # X
        self.train_y = tf.placeholder(tf.int32, [None, ], name="train_y")  # y:[None,num_classes]
        self.test_x = tf.placeholder(tf.int32, [self.sequence_length], name="test_x")

        distances = []
        distances.append(tf.reduce_sum(tf.abs(tf.add(self.train_x, tf.negative(self.test_x))),axis=1))
        index = tf.argmin(distances)
        test_label = self.labels[index]
        return test_label
