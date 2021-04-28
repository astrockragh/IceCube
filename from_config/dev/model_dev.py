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
  
#copy over from other model
class GAT(Model):
    def __init__(self, n_out = 4, hidden_states=64, gat_layers=2, gat_activation='relu', decode_layers=3, decode_activation='relu', regularization=None, dropout=0.2, batch_norm=True, forward=True):
        super().__init__()
        self.n_out=n_out
        self.hidden_states=hidden_states
        self.gat_activation=conv_activation
        self.forward=forward
        self.dropout=dropout
        self.gat_layers=gat_layers
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

class DevEdge(Model):
    def __init__(self, edgeconv, edgenorm, hidden_states=64, edgetype=0, forward=True, K=[1,2], agg_method='min',regularization=None, dropout=0.025):
        super().__init__()
        self.n_out=3
        self.n_sigs=2
        self.hidden_states=hidden_states
        self.conv_activation='relu'
        self.forward=forward
        self.dropout=dropout
        self.Ks=K
        self.agg_method=agg_method
        self.conv_layers=2
        self.decode_layers=2
        self.edgeconv=edgeconv
        self.edgenorm=edgenorm
        self.edgetype=edgetype
        self.regularize=regularization
        self.decode_activation=d_act
        self.batch_norm=True
        # Define layers of the model
        if self.edgenorm:
          self.norm_edge  = BatchNormalization()
        self.MPs      = [SGConv(self.hidden_states, self.hidden_states, K=K, agg_method=self.agg_method, dropout = self.dropout) for K in self.Ks]

        if self.edgeconv:
          self.ECC1    = ECCConv(self.hidden_states, [self.hidden_states, self.hidden_states, self.hidden_states], n_out = self.hidden_states, activation = "relu", kernel_regularizer=self.regularize)

        self.GCNs    = [GraphSageConv(self.hidden_states*int(i), activation=self.conv_activation, kernel_regularizer=self.regularize) for i in 4*2**np.arange(self.conv_layers)]

        self.Pool1   = GlobalMaxPool()
        self.Pool2   = GlobalAvgPool()
        self.Pool3   = GlobalSumPool()

        self.decode  = [Dense(i * self.hidden_states) for i in  2*2**np.arange(self.decode_layers+1,1,-1)]
        self.dropout_layers  = [Dropout(self.dropout) for i in range(len(self.decode))]
        if self.batch_norm:
          self.norm_layers  = [BatchNormalization() for i in range(len(self.decode))]
        else:
          self.norm_layers =  [no_norm for i in range(len(self.decode))]
        
        self.loge     = [Dense(self.hidden_states) for _ in range(2)]
        self.loge_out = Dense(1)
        self.angles     = [Dense(self.hidden_states) for _ in range(2)]
        self.angles_out = Dense(2)
        self.angle_scale= Dense(2)
        if self.n_sigs > 0:
          self.sigs      = [Dense(self.hidden_states) for _ in range(2)]
          self.sigs_out  = Dense(self.n_sigs)

    def call(self, inputs, training = False):
        x, a, i = inputs
        glob_avg=tf.math.segment_mean(x,i)
        glob_var=abs(tf.math.subtract(tf.math.segment_mean(multiply([x,x]),i),multiply([glob_avg, glob_avg])))
        glob_max=tf.math.segment_max(x,i)
        glob_min=tf.math.segment_min(x,i)
        xglob=tf.concat([glob_avg, glob_var, glob_max, glob_min], axis=1)
        a, e    = self.generate_edge_features(x, a)
        if self.edgenorm:
          e=self.norm_edge(e)
        for MP in self.MPs:
          x = MP([x, a, e])
        if self.edgeconv:
          x = self.ECC1([x, a, e])
        for conv in self.GCNs:
          x=conv([x,a]) 
        x1 = self.Pool1([x, i])
        x2 = self.Pool2([x, i])
        x3 = self.Pool3([x, i])
        x = tf.concat([x1, x2, x3], axis = 1)
        x=tf.concat([x, xglob], axis=1)
        for decode_layer, dropout_layer, norm_layer in zip(self.decode, self.dropout_layers, self.norm_layers):
          x = dropout_layer(x, training = training)
          x = self.decode_activation(decode_layer(x))
          x = norm_layer(x, training = training)
                
        x_loge = self.loge[0](x)
        x_loge = self.loge[1](x_loge)
        x_loge = self.loge_out(x_loge)

        x_angles = self.angles[0](x)
        x_angles = self.angles[1](x_angles)
        x_angles = self.angles_out(x_angles)
        zeniazi=sigmoid(self.angle_scale(x_angles))

        if self.n_sigs > 0:
          x_sigs  = self.sigs[0](x)
          x_sigs  = self.sigs[1](x_sigs)
          x_sigs  = tf.abs(self.sigs_out(x_sigs)) + eps
        #could add correlation here  
        xs=tf.stack([x_loge[:,0], zeniazi[:,0]*np.pi, zeniazi[:,1]*2*np.pi], axis = 1)
        if self.n_sigs > 0:
          return tf.concat([xs, x_sigs], axis=1)
        else:
          return xs


    def generate_edge_features(self, x, a):
      send    = a.indices[:, 0]
      receive = a.indices[:, 1]
      
      if self.forward == True: #could maybe be improved
        forwards  = tf.gather(x[:, 3], send) <= tf.gather(x[:, 3], receive)

        send    = tf.cast(send[forwards], tf.int64)
        receive = tf.cast(receive[forwards], tf.int64)

        a       = SparseTensor(indices = tf.stack([send, receive], axis = 1), values = tf.ones(tf.shape(send), dtype = tf.float32), dense_shape = tf.cast(tf.shape(a), tf.int64))
       ##distance vectors
      diff_x  = tf.subtract(tf.gather(x, receive), tf.gather(x, send))

      dists   = tf.sqrt(
        tf.reduce_sum(
          tf.square(
            diff_x[:, :3]
          ), axis = 1
        ))

      vects = tf.math.divide_no_nan(diff_x[:, :3], tf.expand_dims(dists, axis = -1))
      if self.edgetype==0:
        e = tf.concat([diff_x[:, 3:], tf.expand_dims(dists, -1), vects], axis = 1)

      if self.edgetype==1:
    
        ## SRT, could make this is a mask
        prod_x = tf.math.multiply(tf.gather(x, receive), tf.gather(x, send))
        srt = prod_x[:,5]
        e = tf.concat([diff_x[:, 3:], tf.expand_dims(dists, -1), vects, tf.expand_dims(srt, -1)], axis = 1)
      if self.edgetype==2:
        st=2699 # time scale, database specific
        c=tf.constant(0.000299792458) #speed of light in km pr nanosec

        speed = tf.math.divide_no_nan(dists, st*diff_x[:,3]) #could add fudge factor to account for ice c lower than vacuum c
        speed = tf.math.greater_equal(speed, c)
        speed = tf.cast(tf.where(speed, 0,1), tf.float32)
        e = tf.concat([diff_x[:, 3:], tf.expand_dims(dists, -1), vects, tf.expand_dims(speed,-1)], axis = 1)
      if self.edgetype==3:
        #srt comm?
        prod_x = tf.math.multiply(tf.gather(x, receive), tf.gather(x, send))
        srt = prod_x[:,5]
        #higher than c?
        st=2699 # time scale, database specific
        c=tf.constant(0.000299792458) #speed of light in km pr nanosec

        speed = tf.math.divide_no_nan(dists, st*diff_x[:,3]) #could add fudge factor to account for ice c lower than vacuum c
        speed = tf.math.greater_equal(speed, c)
        speed = tf.cast(tf.where(speed, 0,1), tf.float32)
        e = tf.concat([diff_x[:, 3:], tf.expand_dims(dists, -1), vects, tf.expand_dims(srt, -1), tf.expand_dims(speed,-1)], axis = 1)
      return a, e

class MLP(Model):
    def __init__(self, output, hidden=256, layers=2, batch_norm=True,
                 dropout=0.0, activation='relu', final_activation=None):
        super().__init__()
        self.batch_norm = batch_norm
        self.dropout_rate = dropout

        self.mlp = Sequential()
        for i in range(layers):
            # Linear
            self.mlp.add(Dense(hidden if i < layers - 1 else output, activation = activation))
            if dropout > 0:
                self.mlp.add(Dropout(dropout))


    def call(self, inputs, training = False):
        return self.mlp(inputs, training = training)

class SGConv(MessagePassing):
    # note that the D^-1/2 norm is not implemented since it is irrelevant for us 
    def __init__(self, n_out, hidden_states, K=2, agg_method='sum', dropout = 0):

        """Agg_method supports "sum": scatter_sum,
          "mean": scatter_mean,
          "max": scatter_max,
          "min": scatter_min,
          "prod": scatter_prod"""
        super().__init__()
        self.n_out = n_out
        self.agg_method=agg_method
        self.K=K
        self.hidden_states = hidden_states
        self.message_mlps = [MLP(hidden_states * 2, hidden = hidden_states * 4, layers = 2, dropout = dropout) for _ in range(self.K)]
        self.update_mlp  = MLP(hidden_states * 1, hidden = hidden_states * 2, layers = 2, dropout = dropout)


    ##inverted structure since tf requires output func to be propagate
    def prop_khop(self, x, a, k, e=None, training = False, **kwargs):
        self.n_nodes = tf.shape(x)[0]
        self.index_i = a.indices[:, 1]
        self.index_j = a.indices[:, 0]

        # Message
        # print(x, a, e)
        # msg_kwargs = self.get_kwargs(x, a, e, self.msg_signature, kwargs)
        messages = self.message(x, a, k, e, training = training)

        # Aggregate
        # agg_kwargs = self.get_kwargs(x, a, e, self.agg_signature, kwargs)

        ##  make own aggregate
        embeddings = self.aggregate(messages, training = training)

        return embeddings

    def propagate(self, x, a, e, training=False):
        for hop in range(self.K):
          x=self.prop_khop(x,a, hop, e, training = training)
        return self.update(x, training = training)

    def message(self, x, a, k, e, training = False):
        # print([self.get_i(x), self.get_j(x), e])
        out = tf.concat([self.get_i(x), self.get_j(x), e], axis = 1)
        out = self.message_mlps[k](out, training = training)
        return out
    
    def update(self, embeddings, training = False):
        out = self.update_mlp(embeddings, training = training)
        return out