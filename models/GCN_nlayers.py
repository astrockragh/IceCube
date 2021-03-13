import os
import numpy as np
from spektral.layers.convolutional.gcn_conv import GCNConv


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import tensorflow as tf

from spektral.layers import ECCConv
from spektral.layers.pooling.global_pool import GlobalMaxPool, GlobalAvgPool, GlobalSumPool

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.activations import tanh
from tensorflow.sparse import SparseTensor

# Probably needs regularization, but first step is just to fit, then we will regularize.
class model(Model):
    def __init__(self, n_out = 4, hidden_states=64, n_GCN=2, GCN_activation=LeakyReLU(alpha=0.2), decode_activation=LeakyReLU(alpha=0.2), dropout=0.2, forward=True, ECC=True):
        super().__init__()
        self.n_out=n_out
        self.hidden_states=hidden_states
        self.conv_activation=GCN_activation
        self.forward=forward
        self.dropout=dropout
        self.n_GCN=n_GCN
        self.ECC=ECC
        self.decode_activation=decode_activation
        # Define layers of the model
        self.ECC=ECC
        if self.ECC:
          self.ECC1    = ECCConv(hidden_states, [hidden_states, hidden_states, hidden_states], n_out = hidden_states, activation = "relu")
        self.GCNs     = [GCNConv(hidden_states*int(i), activation=GCN_activation) for i in 2**np.arange(n_GCN)]
        self.Pool1   = GlobalMaxPool()
        self.Pool2   = GlobalAvgPool()
        self.Pool3   = GlobalSumPool()
        self.decode  = [Dense(i * hidden_states) for i in  2**np.arange(n_GCN)]
        self.dropout_layers  = [Dropout(dropout) for i in range(len(self.decode))]
        self.norm_layers  = [BatchNormalization() for i in range(len(self.decode))]
        self.final      = Dense(n_out)

    def get_config(self):
      return dict(
          n_out=self.n_out,
          forward=self.forward,
          hidden_states=self.hidden_states,
          conv_activation=self.conv_activation,
          ECCConv=self.ECC,
          n_GCN=self.n_GCN,
          decode_activation=self.decode_activation,
          dropout=self.dropout)

    def call(self, inputs, training = False):
        x, a, i = inputs
        if self.ECC:
          a, e    = self.generate_edge_features(x, a)
          x = self.ECC1([x, a, e])
        for GCN_layer in self.GCNs:
          x=GCN_layer([x,a])
        x1 = self.Pool1([x, i])
        x2 = self.Pool2([x, i])
        x3 = self.Pool3([x, i])
        x = tf.concat([x1, x2, x3], axis = 1)
        for decode_layer, dropout_layer, norm_layer in zip(self.decode, self.dropout_layers, self.norm_layers):
          x = dropout_layer(x, training = training)
          x = self.decode_activation(decode_layer(x))
          x = norm_layer(x, training = training)
        x = self.final(x)
        # tf.print(tf.shape(x))
        return x

    def generate_edge_features(self, x, a):
      send    = a.indices[:, 0]
      receive = a.indices[:, 1]
      
      if self.forward == True:
        forwards  = tf.gather(x[:, 3], send) <= tf.gather(x[:, 3], receive)

        send    = tf.cast(send[forwards], tf.int64)
        receive = tf.cast(receive[forwards], tf.int64)

        a       = SparseTensor(indices = tf.stack([send, receive], axis = 1), values = tf.ones(tf.shape(send), dtype = tf.float32), dense_shape = tf.cast(tf.shape(a), tf.int64))

      diff_x  = tf.subtract(tf.gather(x, receive), tf.gather(x, send))

      dists   = tf.sqrt(
        tf.reduce_sum(
          tf.square(
            diff_x[:, :3]
          ), axis = 1
        ))

      vects = tf.math.divide_no_nan(diff_x[:, :3], tf.expand_dims(dists, axis = -1))

      e = tf.concat([diff_x[:, 3:], tf.expand_dims(dists, -1), vects], axis = 1)

      return a, e