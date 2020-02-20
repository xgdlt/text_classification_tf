# coding=utf-8
#
# created by kpe on 28.Mar.2019 at 12:33
#

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
import numpy as np

class TrigPosEmbedding(keras.layers.Layer):
    """Position embedding use sine and cosine functions.

    See: https://arxiv.org/pdf/1706.03762

    Expand mode:
        # Input shape
            2D tensor with shape: `(batch_size, sequence_length)`.

        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

    Add mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

    Concat mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim + output_dim)`.
    """
    MODE_EXPAND = 'expand'
    MODE_ADD = 'add'
    MODE_CONCAT = 'concat'

    def __init__(self,
                 mode=MODE_ADD,
                 output_dim=None,
                 **kwargs):
        """
        :param output_dim: The embedding dimension.
        :param kwargs:
        """
        if mode in [self.MODE_EXPAND, self.MODE_CONCAT]:
            if output_dim is None:
                raise NotImplementedError('`output_dim` is required in `%s` mode' % mode)
            if output_dim % 2 != 0:
                raise NotImplementedError('It does not make sense to use an odd output dimension: %d' % output_dim)
        self.mode = mode
        self.output_dim = output_dim
        self.supports_masking = True
        super(TrigPosEmbedding, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'mode': self.mode,
            'output_dim': self.output_dim,
        }
        base_config = super(TrigPosEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        if self.mode == self.MODE_EXPAND:
            return input_shape + (self.output_dim,)
        if self.mode == self.MODE_CONCAT:
            return input_shape[:-1] + (input_shape[-1] + self.output_dim,)
        return input_shape

    def call(self, inputs, mask=None):
        input_shape = K.shape(inputs)
        if self.mode == self.MODE_ADD:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], input_shape[2]
            pos_input = K.tile(K.expand_dims(K.arange(seq_len), axis=0), [batch_size, 1])
        elif self.mode == self.MODE_CONCAT:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], self.output_dim
            pos_input = K.tile(K.expand_dims(K.arange(seq_len), axis=0), [batch_size, 1])
        else:
            output_dim = self.output_dim
            pos_input = inputs
        if K.dtype(pos_input) != K.floatx():
            pos_input = K.cast(pos_input, K.floatx())
        evens = K.arange(output_dim // 2) * 2
        odds = K.arange(output_dim // 2) * 2 + 1
        even_embd = K.sin(
            K.dot(
                K.expand_dims(pos_input, -1),
                K.expand_dims(1.0 / K.pow(
                    10000.0,
                    K.cast(evens, K.floatx()) / K.cast(output_dim, K.floatx())
                ), 0)
            )
        )
        odd_embd = K.cos(
            K.dot(
                K.expand_dims(pos_input, -1),
                K.expand_dims(1.0 / K.pow(
                    10000.0, K.cast((odds - 1), K.floatx()) / K.cast(output_dim, K.floatx())
                ), 0)
            )
        )
        embd = K.stack([even_embd, odd_embd], axis=-1)
        output = K.reshape(embd, [-1, K.shape(inputs)[1], output_dim])
        if self.mode == self.MODE_CONCAT:
            output = K.concatenate([inputs, output], axis=-1)
        if self.mode == self.MODE_ADD:
            output += inputs
        return output




class PositionEmbedding(keras.layers.Layer):
    """Turn integers (positions) into dense vectors of fixed size.
    eg. [[-4], [10]] -> [[0.25, 0.1], [0.6, -0.2]]

    Expand mode: negative integers (relative position) could be used in this mode.
        # Input shape
            2D tensor with shape: `(batch_size, sequence_length)`.

        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

    Add mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

    Concat mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim + output_dim)`.
    """
    MODE_EXPAND = 'expand'
    MODE_ADD = 'add'
    MODE_CONCAT = 'concat'

    def __init__(self,
                 input_dim,
                 output_dim,
                 mode=MODE_ADD,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 **kwargs):
        """
        :param input_dim: The maximum absolute value of positions.
        :param output_dim: The embedding dimension.
        :param embeddings_initializer:
        :param embeddings_regularizer:
        :param activity_regularizer:
        :param embeddings_constraint:
        :param mask_zero: The index that represents padding. Only works in `append` mode.
        :param kwargs:
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode = mode
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.embeddings_constraint = keras.constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero is not False

        self.embeddings = None
        super(PositionEmbedding, self).__init__(**kwargs)

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  'mode': self.mode,
                  'embeddings_initializer': keras.initializers.serialize(self.embeddings_initializer),
                  'embeddings_regularizer': keras.regularizers.serialize(self.embeddings_regularizer),
                  'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
                  'embeddings_constraint': keras.constraints.serialize(self.embeddings_constraint),
                  'mask_zero': self.mask_zero}
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.mode == self.MODE_EXPAND:
            self.embeddings = self.add_weight(
                shape=(self.input_dim * 2 + 1, self.output_dim),
                initializer=self.embeddings_initializer,
                name='embeddings',
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
            )
        else:
            self.embeddings = self.add_weight(
                shape=(self.input_dim, self.output_dim),
                initializer=self.embeddings_initializer,
                name='embeddings',
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
            )
        super(PositionEmbedding, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        if self.mode == self.MODE_EXPAND:
            if self.mask_zero:
                output_mask = K.not_equal(inputs, self.mask_zero)
            else:
                output_mask = None
        else:
            output_mask = mask
        return output_mask

    def compute_output_shape(self, input_shape):
        if self.mode == self.MODE_EXPAND:
            return input_shape + (self.output_dim,)
        if self.mode == self.MODE_CONCAT:
            return input_shape[:-1] + (input_shape[-1] + self.output_dim,)
        return input_shape

    def call(self, inputs, **kwargs):
        if self.mode == self.MODE_EXPAND:
            if K.dtype(inputs) != 'int32':
                inputs = K.cast(inputs, 'int32')
            return K.gather(
                self.embeddings,
                K.minimum(K.maximum(inputs, -self.input_dim), self.input_dim) + self.input_dim,
            )
        input_shape = K.shape(inputs)
        if self.mode == self.MODE_ADD:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], input_shape[2]
        else:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], self.output_dim
        pos_embeddings = K.tile(
            K.expand_dims(self.embeddings[:seq_len, :output_dim], axis=0),
            [batch_size, 1, 1],
        )
        if self.mode == self.MODE_ADD:
            return inputs + pos_embeddings
        return K.concatenate([inputs, pos_embeddings], axis=-1)


class BertEmbeddingsLayer(tf.keras.layers.Layer):

    # noinspection PyUnusedLocal
    def __init__(self, vocab_size, hidden_size = 769, embedding_size = None, hidden_dropout = 0.1, use_token_type=True,
                 use_position_embeddings=True,token_type_vocab_size=2,support_masking=True,position_size=512):
        self.vocab_size = vocab_size
        self.use_token_type = use_token_type
        self.use_position_embeddings = use_position_embeddings
        self.token_type_vocab_size = token_type_vocab_size
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.position_size = position_size
        self.embedding_size = embedding_size  # None for BERT, not None for ALBERT

        self.word_embeddings_layer       = None
        self.word_embeddings_2_layer     = None   # for ALBERT
        self.token_type_embeddings_layer = None
        self.position_embeddings_layer   = None
        self.layer_norm_layer = None
        self.dropout_layer    = None

        self.support_masking = support_masking

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        if isinstance(input_shape, list):
            assert len(input_shape) == 2
            input_ids_shape, token_type_ids_shape = input_shape
            self.input_spec = [keras.layers.InputSpec(shape=input_ids_shape),
                               keras.layers.InputSpec(shape=token_type_ids_shape)]
        else:
            input_ids_shape = input_shape
            self.input_spec = keras.layers.InputSpec(shape=input_ids_shape)

        self.word_embeddings_layer = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.hidden_size if self.embedding_size is None else self.embedding_size,
            mask_zero=True,  # =self.mask_zero,
            name="word_embeddings"
        )
        if self.embedding_size is not None:
            # ALBERT word embeddings projection
            self.word_embeddings_2_layer = self.add_weight(name="word_embeddings_2/embeddings",
                                                           shape=[self.embedding_size,
                                                                  self.hidden_size],
                                                           dtype=K.floatx())

        if self.use_token_type:
            self.token_type_embeddings_layer = keras.layers.Embedding(
                input_dim=self.token_type_vocab_size,
                output_dim=self.hidden_size,
                name="token_type_embeddings"
            )
        if self.use_position_embeddings:
            self.position_embeddings_layer = PositionEmbedding(input_dim=self.position_size,output_dim=self.hidden_size)

        self.layer_norm_layer = tf.keras.layers.LayerNormalization(name="LayerNorm")
        self.dropout_layer    = keras.layers.Dropout(rate=self.hidden_dropout)

        super(BertEmbeddingsLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids      = inputs
            token_type_ids = None
        input_ids = tf.cast(input_ids, dtype=tf.int32)

        embedding_output = self.word_embeddings_layer(input_ids)
        if self.word_embeddings_2_layer is not None:  # ALBERT: project embedding to hidden_size
            embedding_output = tf.matmul(embedding_output, self.word_embeddings_2_layer)

        if token_type_ids is not None:
            token_type_ids = tf.cast(token_type_ids, dtype=tf.int32)
            embedding_output += self.token_type_embeddings_layer(token_type_ids)

        if self.position_embeddings_layer is not None:
            embedding_output = self.position_embeddings_layer(embedding_output)

        embedding_output = self.layer_norm_layer(embedding_output)
        embedding_output = self.dropout_layer(embedding_output, training=training)

        return embedding_output   # [B, seq_len, hidden_size]

    def compute_mask(self, inputs, mask=None):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids      = inputs

        # if not self.mask_zero:
        #   return None

        return tf.not_equal(input_ids, 0)




class EmbeddingsLayer(tf.keras.layers.Layer):

    # noinspection PyUnusedLocal
    def __init__(self, config):
        super(EmbeddingsLayer, self).__init__()
        self.config = config
        self.vocab_size =config.vocab_size
        self.use_token_type = config.use_token_type
        self.use_position_embeddings = config.use_position_embeddings
        self.token_type_vocab_size = config.token_type_vocab_size
        self.hidden_size = config.hidden_size
        self.hidden_dropout = config.hidden_dropout
        self.position_size = config.position_size
        self.embedding_size = config.embedding_size  # None for BERT, not None for ALBERT
        self.support_masking = config.support_masking
        self.pretrain_initializer = config.pretrain_initializer

        self.word_embeddings_layer       = None
        self.word_embeddings_2_layer     = None   # for ALBERT
        self.token_type_embeddings_layer = None
        self.position_embeddings_layer   = None
        self.layer_norm_layer = None
        self.dropout_layer    = None


    def build(self, input_shape):
        if isinstance(input_shape, list):
            assert len(input_shape) == 2
            input_ids_shape, token_type_ids_shape = input_shape
            self.input_spec = [keras.layers.InputSpec(shape=input_ids_shape),
                               keras.layers.InputSpec(shape=token_type_ids_shape)]
        else:
            input_ids_shape = input_shape
            self.input_spec = keras.layers.InputSpec(shape=input_ids_shape)

        if self.pretrain_initializer is not None:
            initializer =  tf.constant_initializer(np.load(self.pretrain_initializer))
        else:
            initializer = self.config.initializer

        self.word_embeddings_layer = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.hidden_size if self.embedding_size is None else self.embedding_size,
            embeddings_initializer=initializer,
            name="word_embeddings"
        )
        if self.embedding_size is not None:
            # ALBERT word embeddings projection
            self.word_embeddings_2_layer = self.add_weight(name="word_embeddings_2/embeddings",
                                                           shape=[self.embedding_size,
                                                                  self.hidden_size],
                                                           dtype=K.floatx())

        if self.use_token_type:
            self.token_type_embeddings_layer = keras.layers.Embedding(
                input_dim=self.token_type_vocab_size,
                output_dim=self.hidden_size,
                name="token_type_embeddings"
            )
        if self.use_position_embeddings:
            self.position_embeddings_layer = PositionEmbedding(input_dim=self.position_size,output_dim=self.hidden_size)

        self.layer_norm_layer = tf.keras.layers.LayerNormalization(name="LayerNorm")
        self.dropout_layer    = keras.layers.Dropout(rate=self.hidden_dropout)

        super(EmbeddingsLayer, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids      = inputs
            token_type_ids = None
        input_ids = tf.cast(input_ids, dtype=tf.int32)

        embedding_output = self.word_embeddings_layer(input_ids)
        if self.word_embeddings_2_layer is not None:  # ALBERT: project embedding to hidden_size
            embedding_output = tf.matmul(embedding_output, self.word_embeddings_2_layer)

        if token_type_ids is not None:
            token_type_ids = tf.cast(token_type_ids, dtype=tf.int32)
            embedding_output += self.token_type_embeddings_layer(token_type_ids)

        if self.position_embeddings_layer is not None:
            embedding_output = self.position_embeddings_layer(embedding_output)

        embedding_output = self.layer_norm_layer(embedding_output)
        embedding_output = self.dropout_layer(embedding_output, training=training)

        return embedding_output   # [B, seq_len, hidden_size]

    def compute_mask(self, inputs, mask=None):
        if isinstance(inputs, list):
            assert 2 == len(inputs), "Expecting inputs to be a [input_ids, token_type_ids] list"
            input_ids, token_type_ids = inputs
        else:
            input_ids      = inputs

        return tf.not_equal(input_ids, 0)


def read_npy_file(npyfile):
    data = np.load(npyfile)
    return data.astype(np.float32)
