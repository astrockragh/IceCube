import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.random import set_seed
from spektral.transforms.layer_preprocess import LayerPreprocess
from spektral.layers import GATConv
from spektral.layers.pooling.global_pool import GlobalMaxPool, GlobalAvgPool, GlobalSumPool
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization
from tensorflow.keras.activations import tanh
from tensorflow.sparse import SparseTensor

hidden_states = 16
activation = LeakyReLU(alpha = 0.1)

class model(Model):
    def __init__(self, n_out = 3):
        super().__init__()
        # Define layers of the model
        self.att1 = GATConv(hidden_states, attn_heads=2, dropout_rate=0.4, activation = "relu", return_attn_coef=True) #required keywords is channels/hidden states
        self.att2 = GATConv(hidden_states//2, attn_heads=3, dropout_rate=0.1, activation = "relu")# attn heads are the time limiting key_word, watch out with it
        self.att3 = GATConv(hidden_states*2, attn_heads=4, dropout_rate=0.7, activation = "relu")  # hiddenstates has to be pretty low as well
        self.Pool1   = GlobalAvgPool() #good results with all three
        self.Pool2   = GlobalSumPool()
        self.Pool3   = GlobalMaxPool() #important for angle fitting
        self.decode  = [Dense(size * hidden_states) for size in [16, 8, 4]]
        self.norm_layers  = [BatchNormalization() for i in range(len(self.decode))]
        self.d2      = Dense(n_out)

    def call(self, inputs, training = False):
        x, a, i = inputs
        # a=sp_matrix_to_sp_tensor(a)
        LayerPreprocess(self.att1)
        LayerPreprocess(self.att2)
        x, alpha = self.att1([x,a])
        x = self.att2([x, a])
        x = self.att3([x,a])
        x1 = self.Pool1([x, i])
        x2 = self.Pool2([x, i])
        x3 = self.Pool3([x,i])
        x = tf.concat([x1, x2, x3], axis = 1)
        # x = tf.concat([x1, x2], axis = 1)
        # x = tf.concat([x2, x3], axis = 1)
        for decode_layer, norm_layer in zip(self.decode, self.norm_layers):
          x = activation(decode_layer(x))
          x = norm_layer(x, training = training)
        x = self.d2(x)
        return x, alpha