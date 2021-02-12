##zeroth version of a GCN with the most basic of layers

import os
from spektral.layers.convolutional.gcn_conv import GCNConv


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import tensorflow as tf

from spektral.layers.pooling.global_pool import GlobalMaxPool, GlobalAvgPool, GlobalSumPool

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.activations import tanh
from tensorflow.sparse import SparseTensor


activation = LeakyReLU(alpha = 0.1)

# Needs regularization

class GCN0(Model):
    '''Pretty shallow beginner model but let's see if we can get something going!'''
    def __init__(self, n_labels, hidden_states=64, activation='relu', output_activation="softmax", use_bias=False, dropout_rate=0.5, n_input_channels=None,**kwargs,):
        super().__init__(**kwargs)

        self.n_labels = n_labels
        self.hidden_states=hidden_states
        self.activation = activation
        self.output_activation = output_activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.n_input_channels = n_input_channels
        ## need to add some regularization terms
        self._d0 = Dropout(dropout_rate)
        # this can hold a kernel_regularizer=reg which is yet to be implemented
        self._gcn0 = GCNConv(
            hidden_states, activation=activation, use_bias=use_bias
        )
        self.d1 = Dropout(dropout_rate)
        self.gcn1 = GCNConv(
            n_labels, activation=output_activation, use_bias=use_bias
        )

        # if tf.version.VERSION < "2.2":
        #     if n_input_channels is None:
        #         raise ValueError("n_input_channels required for tf < 2.2")
        #     x = tf.keras.Input((n_input_channels,), dtype=tf.float32)
        #     a = tf.keras.Input((None,), dtype=tf.float32, sparse=True)
        #     self._set_inputs((x, a))

    def get_config(self):
      # when regularization is added it should go here
        return dict(
            n_labels=self.n_labels,
            hidden_states=self.hidden_states,
            activation=self.activation,
            output_activation=self.output_activation,
            use_bias=self.use_bias,
            dropout_rate=self.dropout_rate,
            n_input_channels=self.n_input_channels,
        )
    def call(self, inputs, training=False):
        x, a = inputs
        if self.n_input_channels is None:
            self.n_input_channels = x.shape[-1]
        else:
            assert self.n_input_channels == x.shape[-1]
        x = self._d0(x)
        x = self._gcn0([x, a])
        x = self._d1(x)
        return self._gcn1([x, a])