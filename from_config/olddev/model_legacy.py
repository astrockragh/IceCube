import os
import numpy as np
from spektral.layers.convolutional.gcn_conv import GCNConv


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import tensorflow as tf

from spektral.layers import ECCConv, GraphSageConv, MessagePassing
from spektral.layers.pooling.global_pool import GlobalMaxPool, GlobalAvgPool, GlobalSumPool

from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout, multiply
from tensorflow.keras.activations import tanh, sigmoid
from tensorflow.sparse import SparseTensor

eps=1e-5

print('loading model')

d_act=LeakyReLU(alpha=0.15)

def no_norm(x, training):
  return x

class GCN_nlayers(Model):
    def __init__(self, n_out = 4, hidden_states=64, conv_layers=2, conv_activation='relu', decode_layers=2, decode_activation='relu', regularization=None, dropout=0.2, batch_norm=True, forward=True, edgeconv=True):
        super().__init__()
        self.n_out=n_out
        self.hidden_states=hidden_states
        self.conv_activation=conv_activation
        self.forward=forward
        self.dropout=dropout
        self.conv_layers=conv_layers
        self.edgeconv=edgeconv
        self.regularize=regularization
        if type(decode_activation)==str:
          self.decode_activation=tf.keras.activations.get(decode_activation)
        else:
          self.decode_activation=decode_activation
        self.batch_norm=batch_norm
        # Define layers of the model
        if self.edgeconv:
          self.ECC1    = ECCConv(hidden_states, [hidden_states, hidden_states, hidden_states], n_out = hidden_states, activation = "relu", kernel_regularizer=self.regularize)
        self.GCNs     = [GCNConv(hidden_states*int(i), activation=self.conv_activation, kernel_regularizer=self.regularize) for i in 2**np.arange(self.conv_layers)]
        self.Pool1   = GlobalMaxPool()
        self.Pool2   = GlobalAvgPool()
        self.Pool3   = GlobalSumPool()
        self.decode  = [Dense(i * hidden_states, activation=self.decode_activation) for i in  2**np.arange(decode_layers)]
        self.dropout_layers  = [Dropout(dropout) for i in range(len(self.decode))]
        if self.batch_norm:
          self.norm_layers  = [BatchNormalization() for i in range(len(self.decode))]
        else:
          self.norm_layers =  [no_norm for i in range(len(self.decode))]
        self.final      = Dense(n_out)
        self.angle_scale= Dense(2)

    def call(self, inputs, training = False):
        x, a, i = inputs
        if self.edgeconv:
          a, e    = self.generate_edge_features(x, a)
          x = self.ECC1([x, a, e])
        for GCN_layer in self.GCNs:
          x=GCN_layer([x,a])
        x1 = self.Pool1([x, i])
        x2 = self.Pool2([x, i])
        x3 = self.Pool3([x, i])
        # tf.print(x1,x2,x3, x1.shape, x2.shape, x3)
        x = tf.concat([x1, x2, x3], axis = 1)
        for decode_layer, dropout_layer, norm_layer in zip(self.decode, self.dropout_layers, self.norm_layers):
          x = dropout_layer(x, training = training)
          x = self.decode_activation(decode_layer(x))
          x = norm_layer(x, training = training)
        x = self.final(x)

        zeniazi=x[:,1:3]
        zeniazi=sigmoid(self.angle_scale(zeniazi))
        # tf.print(x)
        x1=tf.stack([x[:,0],zeniazi[:,0]*np.pi, zeniazi[:,1]*2*np.pi], axis=1)
        x=tf.concat([x1, x[:,3:]], axis=1)
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