import os, sys, argparse, importlib, time, inspect
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
if hasattr(__builtins__,'__IPYTHON__'):
    print('Notebook')
    from tqdm.notebook import tqdm
else:
    print('Not notebook')
    from tqdm import tqdm
from tensorflow.keras import backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow_probability as tfp

gpu_devices = tf.config.list_physical_devices('GPU') 
if len(gpu_devices) > 0:
    print("GPU detected")
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print('No GPU detected')

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model, Model
import spektral

from spektral.data import DisjointLoader, BatchLoader, SingleLoader
from importlib import reload
import winsound
import datetime as dt

################################################
# Setup Deafult Variables and Hyperparams    #   #this should be made into a loadable JSON
################################################

# scenarios=[]
# for scenario in scenarios:

model = 
learning_rate = 5e-4
batch_size    = 512
epochs        = 20
n_data       = 1e5
scenario    = "hpc_test"
patience = 5
wandblog=0

hidden_states = 'N/A'
forward       = False
dropout       = 'None'
loss_method   = "loss_func_linear_angle"
n_neighbors   = 6 # SKRIV SELV IND

if wandblog:
    import wandb
    !wandb login b5b917a9390932e56fccfcbff6f528ccd85c44bf

import data_load as dl
graph_data=dl.graph_data
dataset=graph_data(n_data=n_data, restart=1, pos=1)
idx_lists = dataset.index_lists
# Split data
dataset_train = dataset[idx_lists[0]]
dataset_val   = dataset[idx_lists[1]]
dataset_test  = dataset[idx_lists[2]]