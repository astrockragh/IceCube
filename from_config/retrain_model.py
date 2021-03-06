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

os.environ["CUDA_VISIBLE_DEVICES"]="0"

gpu_devices = tf.config.list_physical_devices('GPU') 
if len(gpu_devices) > 0:
    print("GPU detected")
    for i in range(len(gpu_devices)):
    	tf.config.experimental.set_memory_growth(gpu_devices[i], True)
else:
    print('No GPU detected')

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model, Model
import spektral
from sklearn.preprocessing import normalize
from spektral.data import DisjointLoader, BatchLoader, SingleLoader
from importlib import reload
# import winsound
import wandb
import datetime as dt
wandblog=1

################################################
# Setup Deafult Variables                       # 
################################################
learning_rate = 1e-4
batch_size    = 1024
epochs        = 10
scenario    = "2d->3d"
patience = 20

################################################
# Setup Hyperparameters                        # 
################################################
loss_method   = "abs_vonMises_angle2"

with tf.device('/gpu:0'):
    save_options = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    model=tf.keras.models.load_model("trained_models/IceCube_neutrino/KHop_base_max_3fsagz9i", options=save_options)
model.compile()

def get_metrics(metric_name):
    # Returns a list of functions
    import dev.metrics as metrics
    metrics=getattr(metrics, metric_name)
    return metrics


def get_loss_func(name):
    # Return loss func from the loss functions file given a function name
    import dev.loss_funcs as loss_func_module
    loss_func = getattr(loss_func_module, name)
    return loss_func


def get_performance(name):
    # Return performance plotter from the eval_model file given a name
    import dev.eval_model as evals
    performance_plot = getattr(evals, name)
    return performance_plot 


################################################
# Load data                      # 
################################################
data_params={ "n_steps": 10,
        "graph_construction":       "classic",
        "muon":             False,
        "n_neighbors":       31,
        "restart":   False,
        "transform_path": "../db_files/dev_lvl7/transformers.pkl",
        "db_path": "../db_files/dev_lvl7/dev_lvl7_mu_nu_e_classification_v003.db",
        "features":   ["dom_x", "dom_y", "dom_z", "dom_time", "charge_log10", "width", "rqe"],
        "targets":    ["energy_log10", "zenith","azimuth","event_no"],
        "database": "submit"
    }
import dev.testtraindata as dl
reload(dl)
graph_data=dl.graph_data
dataset_train=graph_data(**data_params, traintest='train')
dataset_test=graph_data(**data_params, traintest='mix')

dataset_val=dataset_test

loader_train = DisjointLoader(dataset_train, epochs=epochs, batch_size=batch_size) # the different loaders work very very differently, beware
loader_test = DisjointLoader(dataset_test, batch_size=batch_size, epochs=1)

loss_func             = get_loss_func(loss_method)
metrics               = get_metrics('energy_angle_zeniazi')
performance_plot      = get_performance("performance_vM2D")
import dev.lr_schedules as lr_module

lr_generator = getattr(lr_module, 'classic')

lr_schedule  = lr_generator(1e-5, 0, 0.95)()


if wandblog:
    import wandb
    run = wandb.init(project = 'IceCube_neutrino', entity = "chri862z", group='new_loss', reinit=True, settings=wandb.Settings(start_method="fork"))
    wandb.run.name = scenario+'_'+str(wandb.run.id)

# Define training function
@tf.function(input_signature = loader_test.tf_signature(), experimental_relax_shapes = True)
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


def validation(loader):
    loss = 0
    prediction_list, target_list = [], []
    for batch in loader:
        inputs, targets = batch
        predictions, targets, out = test_step(inputs, targets)
        loss           += out
        
        prediction_list.append(predictions)
        target_list.append(targets)
    
    y_reco  = tf.concat(prediction_list, axis = 0)
    y_true  = tf.concat(target_list, axis = 0)
    y_true  = tf.cast(y_true, tf.float32)

    loss, loss_from = loss_func(y_reco, y_true, re=True)
    
    energy, e_old, alpha, zeni, azi= metrics(y_reco, y_true)
    
    return loss, loss_from, [energy, e_old, alpha, zeni, azi]

n_steps=10
model_path='trained_models/IceCube_neutrino/'+wandb.run.name
save_path=model_path

loader_train = DisjointLoader(dataset_train, epochs=epochs, batch_size=batch_size)

if not osp.isdir(save_path):
    os.makedirs(save_path)
    print('New folder for saving run made')

    # Learning rate and optimizer
    learning_rate            = next(lr_schedule)
    opt           = Adam(learning_rate)

k=2

steps_per_epoch=loader_train.steps_per_epoch//k
val_epoch=1
tot_time=0
current_batch = 0
current_epoch = 1
loss          = 0
lowest_loss   = np.inf
early_stop    = 1
early_stop_counter    = 0
pbar          = tqdm(total = steps_per_epoch*n_steps, position=0, leave = True)
start_time    = time.time()
summarylist=[]
for j in range(epochs):
    for i in range(n_steps):
        # if n_steps!=10:
        #     i_t=np.random.randint(0,10)
        # else:
        #     i_t=i
        start=time.time()
        dataset_train=graph_data(**data_params, traintest='train', i_train=i)
        stop=time.time()
        print(f'Loaded dataset in {np.round(stop-start, 2)} s')
        loader_train = DisjointLoader(dataset_train[::k], epochs=1, batch_size=batch_size)
        loader_train=loader_train.load().prefetch(1)
        for batch in loader_train:
            inputs, targets = batch
            out             = train_step(inputs, targets)
            loss           += out
            if current_epoch==1 and current_batch==0:
                model.summary()
                if wandblog:
                    summary=model.summary(print_fn=summarylist.append)
                    table=wandb.Table(columns=["Layers"])
                    for s in summarylist:
                        table.add_data(s)
                    wandb.log({'Model summary': table})
            current_batch  += 1
            pbar.update(1)
            pbar.set_description(f"Epoch {current_epoch} / {epochs}; Avg_loss: {loss / current_batch:.6f}")
            
            # if current_batch == steps_per_epoch:
            if current_batch%steps_per_epoch-1==0:
                t=time.time() - start_time
                tot_time+=t
                print(f"Step {i} of Epoch {current_epoch} of {epochs} done in {t:.2f} seconds using learning rate: {learning_rate:.2E}")
                print(f"Avg loss of train: {loss / (steps_per_epoch*n_steps):.6f}")

                loader_val    = DisjointLoader(dataset_val[::2], epochs = 1,      batch_size = batch_size)
                val_loss, val_loss_from, val_metric = validation(loader_val)
                if wandblog:
                    wandb.log({"Train Loss":      loss / (steps_per_epoch*n_steps),
                            "Validation Loss": val_loss, 
                            "w(log(E))":   val_metric[1],
                            "Energy bias":   val_metric[0][1],
                            "Energy sig-1":   val_metric[0][0],
                            "Energy sig+1":   val_metric[0][2],
                            "Solid angle 68th":    val_metric[2][3],
                            "Angle bias":   val_metric[2][1],
                            "Angle sig-1":   val_metric[2][0],
                            "Angle sig+1":   val_metric[2][2],
                            "zenith 68th":    val_metric[3][3],
                            "zenith bias":   val_metric[3][1],
                            "zenith sig-1":   val_metric[3][0],
                            "zenith sig+1":   val_metric[3][2],
                            "azimuth 68th":    val_metric[4][3],
                            "azimuth bias":   val_metric[4][1],
                            "azimuth sig-1":   val_metric[4][0],
                            "azimuth sig+1":   val_metric[4][2],
                            "Learning rate":   learning_rate})
                print("\n")                 
                print(f"Avg loss of validation: {val_loss:.6f}")
                # print(f"Loss from:  Energy: {val_loss_from[0]:.6f} \t Zenith: {val_loss_from[1]:.6f} \t Azimuth {val_loss_from[2]:.6f}")
                print(f"Loss from:  Energy: {val_loss_from[0]:.6f} \t Angle: {val_loss_from[1]:.6f}")
                print(f"Energy: bias = {val_metric[0][1]:.6f} sig_range = {val_metric[0][0]:.6f}<->{val_metric[0][2]:.6f}, old metric {val_metric[1]:.6f}\
                    \n Angle: bias = {val_metric[2][1]:.6f} sig_range = {val_metric[2][0]:.6f}<->{val_metric[2][2]:.6f}, old metric {val_metric[2][3]:.6f}\
                    \n Zenith: bias = {val_metric[3][1]:.6f} sig_range = {val_metric[3][0]:.6f}<->{val_metric[3][2]:.6f}, old metric {val_metric[3][3]:.6f}\
                    \n Azimuth: bias = {val_metric[4][1]:.6f} sig_range = {val_metric[4][0]:.6f}<->{val_metric[4][2]:.6f}, old metric {val_metric[4][3]:.6f}")

              
                if val_loss < lowest_loss:
                    early_stop_counter = 0
                    lowest_loss        = val_loss
                else:
                    early_stop_counter += 1
                print(f'Early stop counter: {early_stop_counter}/{patience}, lowest val loss was {lowest_loss:.6f}')
                if early_stop and (early_stop_counter >= patience):
                    model.save(save_path)
                    print(f"Stopped training. No improvement was seen in {patience} epochs")


                if current_epoch % val_epoch == 0:
                    if wandblog:
                        loader_test = DisjointLoader(dataset_test, batch_size=batch_size, epochs=1)
                        fig, _ = performance_plot(loader_test, test_step, metrics, bins=20, save=True, save_path=save_path)
                        title="performanceplot_"+str(current_epoch)+f'_{i}'
                        wandb.log({title: [wandb.Image(fig, caption=title)]})
                if current_batch==steps_per_epoch*n_steps:
                    learning_rate = next(lr_schedule)
                    opt.learning_rate.assign(learning_rate)
                    loss            = 0
                    start_time      = time.time()
                    current_epoch  += 1
                    current_batch   = 0
                    # if current_epoch != epochs:
                    pbar          = tqdm(total = steps_per_epoch*n_steps, position=0, leave = True)
                    model.save(save_path)
                    print("Model saved")

# fig, ax = test_angle(loader_test)
# if wandblog:
#     fig.savefig(f"model_tests/{scenario}_test.pdf")


