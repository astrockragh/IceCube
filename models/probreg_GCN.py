import os
from spektral.layers.convolutional.gcn_conv import GCNConv


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import tensorflow as tf

from spektral.layers import ECCConv
from spektral.layers.pooling.global_pool import GlobalMaxPool, GlobalAvgPool, GlobalSumPool

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
from tensorflow.keras.activations import tanh
from tensorflow.sparse import SparseTensor
from tensorflow.keras.backend import epsilon


hidden_states = 64
activation = LeakyReLU(alpha = 0.15)
eps = 1e-5

# Probably needs regularization, but first step is just to fit, then we will regularize.

class model(Model):
    def __init__(self, n_out = 6, hidden_states = 64, forward = False, dropout = 0.5):
        super().__init__()
        self.forward = forward
        # Define layers of the model
        self.ECC1    = ECCConv(hidden_states, [hidden_states, hidden_states], n_out = hidden_states, activation = "relu")
        self.GCN1    = GCNConv(hidden_states, activation = "relu")
        self.GCN2    = GCNConv(hidden_states * 2, activation = "relu")
        self.GCN3    = GCNConv(hidden_states * 4, activation = "relu")
        self.GCN4    = GCNConv(hidden_states * 8, activation = "relu")
        self.Pool1   = GlobalMaxPool()
        self.Pool2   = GlobalAvgPool()
        # self.Pool3   = GlobalSumPool()
        self.decode  = [Dense(size * hidden_states) for size in [16, 16, 8]]
        self.drop_w  = [Dropout(dropout) for _ in range(len(self.decode))]
        self.norm_layers  = [BatchNormalization() for _ in range(len(self.decode))]
        self.angles     = [Dense(hidden_states) for _ in range(2)]
        self.angles_out = Dense(2)
        self.sigs      = [Dense(hidden_states) for _ in range(2)]
        self.sigs_out  = Dense(2)

        #consider modifying it to contain full covariance, not just diag
        # self.rhos      = Dense(hidden_states // 2)
        # self.rhos_out  = Dense(3)


    def call(self, inputs, training = False):
        x, a, i = inputs
        a, e    = self.generate_edge_features(x, a)
        x = self.ECC1([x, a, e])
        x = self.GCN1([x, a])
        x = self.GCN2([x, a])
        x = self.GCN3([x, a])
        x = self.GCN4([x, a])
        x1 = self.Pool1([x, i])
        x2= self.Pool2([x, i])
        # x3 = self.Pool3([x, i])
        # x = tf.concat([x1, x2, x3], axis = 1)
        x = tf.concat([x1, x2], axis = 1)
        for decode_layer, norm_layer, drop_w in zip(self.decode, self.norm_layers, self.drop_w):
          x = drop_w(x, training = training)
          x = activation(decode_layer(x))
          x = norm_layer(x, training = training)

        x_units = self.angles[0](x)
        x_units = self.angles[1](x_units)
        x_units = self.angles_out(x_units)
        x_units = tf.math.divide_no_nan(x_units, tf.expand_dims(tf.math.reduce_euclidean_norm(x_units, axis = 1), axis = -1))

        x_sigs  = self.sigs[0](x)
        x_sigs  = self.sigs[1](x_sigs)
        x_sigs  = tf.abs(x_sigs) + eps #add small to avoid non-invertability

        # x_rhos  = self.rhos(tf.concat([x, x_units, x_sigs], axis = 1))
        # x_rhos  = tanh(self.rhos_out(x_rhos))

        return tf.concat([x_units, x_sigs], axis = 1)


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