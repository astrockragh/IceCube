import os
from spektral.layers.convolutional import GCSConv


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import tensorflow as tf

from spektral.layers import ECCConv
from spektral.layers.pooling.global_pool import GlobalMaxPool, GlobalAvgPool, GlobalSumPool

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.activations import tanh
from tensorflow.sparse import SparseTensor


hidden_states = 64
activation = LeakyReLU(alpha = 0.1)

# Probably needs regularization, but first step is just to fit, then we will regularize.

class model(Model):
    def __init__(self, n_out = 7):
        super().__init__()
        # Define layers of the model
        self.ECC1    = ECCConv(hidden_states, [hidden_states, hidden_states, hidden_states], n_out = hidden_states, activation = "relu")
        self.GCS1    = GCSConv(hidden_states, activation = "relu")
        self.GCS2    = GCSConv(hidden_states * 2, activation = "relu")
        self.GCS3    = GCSConv(hidden_states * 4, activation = "relu")
        self.GCS4    = GCSConv(hidden_states * 8, activation = "relu")
        self.Pool1   = GlobalMaxPool()
        self.Pool2   = GlobalAvgPool()
        self.Pool3   = GlobalSumPool()
        self.decode  = [Dense(size * hidden_states) for size in [16, 8, 4, 2, 2]]
        self.norm_layers  = [BatchNormalization() for i in range(len(self.decode))]
        self.d2      = Dense(n_out)

    def call(self, inputs, training = False):
        x, a, i = inputs
        a, e    = self.generate_edge_features(x, a)
        x = self.ECC1([x, a, e])
        x = self.GCS1([x, a])
        x = self.GCS2([x, a])
        x = self.GCS3([x, a])
        x = self.GCS4([x, a])
        x1 = self.Pool1([x, i])
        x2 = self.Pool2([x, i])
        x3 = self.Pool3([x, i])
        x = tf.concat([x1, x2, x3], axis = 1)
        for decode_layer, norm_layer in zip(self.decode, self.norm_layers):
          x = activation(decode_layer(x))
          x = norm_layer(x, training = training)
        x = self.d2(x)
        # tf.print(tf.shape(x))
        return x

    def generate_edge_features(self, x, a):
      send    = a.indices[:, 0]
      receive = a.indices[:, 1]

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
