import os, sys, time
from tqdm import tqdm
import tensorflow as tf
import numpy as np

import os.path as osp

from tensorflow.keras.optimizers import Adam
from spektral.data import DisjointLoader
from importlib import __import__

cwd = osp.abspath('')

def test_data(construct_dict):
    """
    Train a model given a construction dictionairy
    """

    # Setup Log 
    wandblog=construct_dict["wandblog"]
    if wandblog:
        import wandb
        run = wandb.init(project = 'datagen', entity = "chri862z", group=construct_dict["group"], config = construct_dict, reinit=True, settings=wandb.Settings(start_method="fork"))
        wandb.run.name = construct_dict['model_name']+'_'+construct_dict['experiment_name']+'_'+str(wandb.run.id)
    

    import dev.datawhere as dl
    graph_data=dl.graph_data
    dataset_train=graph_data(**construct_dict['data_params'], traintest='train', i_train=construct_dict['data_params']['n_steps']-1)
    dataset_test=graph_data(**construct_dict['data_params'], traintest='test', i_test=construct_dict['data_params']['n_steps']-1)
    dataset_val=dataset_test
    batch_size=512
    
  
    print('Loaded datasets')
    

    loader_train = DisjointLoader(dataset_train, epochs=1, batch_size=batch_size)
    loader_test = DisjointLoader(dataset_test, batch_size=batch_size, epochs=1)


        # Define training function
    @tf.function(input_signature = loader_train.tf_signature(), experimental_relax_shapes = True)
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training = True)
            targets     = tf.cast(targets, tf.float32)
            loss        = loss_func(predictions, targets)
            loss       += sum(model.losses)

        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function(input_signature = loader_test.tf_signature(), experimental_relax_shapes = True)
    def test_step(inputs, targets):
        predictions = model(inputs, training = False)
        targets     = tf.cast(targets, tf.float32) 
        out         = loss_func(predictions, targets)

        return predictions, targets, out
    print("Data generated, everything looks good!")
    return 1
    ################################################



