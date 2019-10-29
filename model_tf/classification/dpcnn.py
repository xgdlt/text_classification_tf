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
 
from dataset.classification_dataset import ClassificationDataset as cDataset
from model_tf.classification.classifier import Classifier


class DPCNN(keras.Model):
    """
    Reference:
        Deep Pyramid Convolutional Neural Networks for Text Categorization
    """

    def __init__(self,  config):
        super(DPCNN, self).__init__(config)
        self.embedding = keras.layers.Embedding(config.DPCNN.input_dim, config.DPCNN.embedding_dimension,
                                                input_length=config.DPCNN.input_length)
        self.dropout1d = keras.layers.SpatialDropout1D(config.DPCNN.dropout)
        self.conv_1 = keras.layers.Conv1D(filters=config.DPCNN.filters, kernel_size=1, padding='SAME', \
                                          kernel_regularizer=keras.regularizers.l2(config.DPCNN.l2), activation="prelu")


    def get_parameter_optimizer_dict(self):
        params = super(DPCNN, self).get_parameter_optimizer_dict()
        params.append({'params': self.convert_conv.parameters()})
        params.append({'params': self.convs.parameters()})
        params.append({'params': self.linear.parameters()})
        return params

    def forward(self, batch):
        if self.config.feature.feature_names[0] == "token":
            embedding = self.token_embedding(
                batch[cDataset.DOC_TOKEN].to(self.config.device))
        else:
            embedding = self.char_embedding(
                batch[cDataset.DOC_CHAR]).to(self.config.device)
        embedding = embedding.permute(0, 2, 1)
        conv_embedding = self.convert_conv(embedding)
        conv_features = self.convs[0](conv_embedding)
        conv_features = conv_embedding + conv_features
        for i in range(1, len(self.convs)):
            block_features = F.max_pool1d(
                conv_features, self.kernel_size, self.pooling_stride)
            conv_features = self.convs[i](block_features)
            conv_features = conv_features + block_features
        doc_embedding = F.max_pool1d(
            conv_features, conv_features.size(2)).squeeze()
        return self.dropout(self.linear(doc_embedding))
