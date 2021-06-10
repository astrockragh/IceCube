import os, sys, time
from tqdm import tqdm
import tensorflow as tf
import numpy as np

import os.path as osp

from tensorflow.keras.optimizers import Adam
# import tensorflow_probability.distributions as tfd
from spektral.data import DisjointLoader
from importlib import __import__

cwd = osp.abspath('')

def train_kde(construct_dict):
    """
    Train a model given a construction dictionairy
    """

    # Setup Log 
    wandblog=construct_dict["wandblog"]
    if wandblog:
        import wandb
        run = wandb.init(project = construct_dict["experiment"], entity = "chri862z", group=construct_dict["group"], config = construct_dict, reinit=True, settings=wandb.Settings(start_method="fork"))
        wandb.run.name = construct_dict['model_name']+'_'+construct_dict['experiment_name']+'_'+str(wandb.run.id)
    
    ################################################
    #   Load dataset                              #
    ################################################
    # import dev.submit_traindata as dl
    # # reload(dl)
    # dataset_train=dl.graph_data(**construct_dict['data_params'])

    from dev.loss_funcs import kde

    import dev.datawhere as dl
    graph_data=dl.graph_data
    dataset_train=graph_data(**construct_dict['data_params'], traintest='train')
    dataset_test=graph_data(**construct_dict['data_params'], traintest='test')
    dataset_val   = dataset_test



    zenith=[]
    print('Making kde true')
    for i in range(len(dataset_val)):
        zenith.append(dataset_val[i].y[1])
    X=np.array(zenith).reshape(-1,1)
    X=tf.convert_to_tensor(X.flatten(), tf.float32)
    kdet=kde(X)

    epochs      = int(construct_dict['run_params']['epochs'])
    batch_size  = int(construct_dict['run_params']['batch_size'])
    
  
    print('Loaded datasets')
    

    loader_train = DisjointLoader(dataset_train, epochs=1, batch_size=batch_size)
    loader_test = DisjointLoader(dataset_test, batch_size=batch_size, epochs=1)


     ###############################################
    #   Setup other run params                     #
    ################################################

    early_stop  = construct_dict['run_params']['early_stop']
    patience    = construct_dict['run_params']['patience']
    val_epoch = construct_dict['run_params']['val_epoch']

    print('check')
    ################################################
    #   Setup model, loss, lr schedule and metrics #
    ################################################

    # Get model, metrics, lr_schedule and loss function

    model, model_path     = setup_model(construct_dict)
        
    loss_func             = get_loss_func(construct_dict['run_params']['loss_func'])
    metrics               = get_metrics(construct_dict['run_params']['metrics'])
    performance_plot      = get_performance(construct_dict['run_params']['performance_plot'])
    lr_schedule          = get_lr_schedule(construct_dict)
    save_path=osp.join(model_path,wandb.run.name)

    if not osp.isdir(save_path):
        os.makedirs(save_path)
        print('New folder for saving run made')

    # Learning rate and optimizer
    learning_rate            = next(lr_schedule)
    opt           = Adam(learning_rate)

    ################################################
    #   Set up TF functions and validation step   #
    ################################################


    # Define training function
    @tf.function(input_signature = loader_train.tf_signature(), experimental_relax_shapes = True)
    def train_step(inputs, targets):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training = True)
            targets     = tf.cast(targets, tf.float32)
            loss        = loss_func(predictions, targets, kdet)
            loss       += sum(model.losses)

        gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function(input_signature = loader_test.tf_signature(), experimental_relax_shapes = True)
    def test_step(inputs, targets):
        predictions = model(inputs, training = False)
        targets     = tf.cast(targets, tf.float32) 
        out         = loss_func(predictions, targets, kdet)

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

        loss, loss_from = loss_func(y_reco, y_true, kdet, re=True)
        
        energy, e_old, alpha, zeni, azi= metrics(y_reco, y_true)
        
        return loss, loss_from, [energy, e_old, alpha, zeni, azi]



    ################################################
    #  Train Model                                 #      
    ################################################
    import gc
    n_steps=construct_dict['data_params']['n_steps']
    steps_per_epoch=loader_train.steps_per_epoch
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
            del dataset_train
            del loader_train
            gc.collect()
            dataset_train=graph_data(**construct_dict['data_params'], traintest='train', i_train=i)
            loader_train = DisjointLoader(dataset_train, epochs=1, batch_size=batch_size)
            # loader_train=loader_train.load().prefetch(1)
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
                
                if current_batch == steps_per_epoch*n_steps:
                # if current_batch == :
                    t=time.time() - start_time
                    tot_time+=t
                    print(f"Epoch {current_epoch} of {epochs} done in {t:.2f} seconds using learning rate: {learning_rate:.2E}")
                    print(f"Avg loss of train: {loss / (steps_per_epoch*n_steps):.6f}")

                    loader_val    = DisjointLoader(dataset_val, epochs = 1,      batch_size = batch_size)
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

                    if construct_dict['run_params']['print_metric']=='zeniaziangle':
                        print(f"Avg loss of validation: {val_loss:.6f}")
                        print(f"Loss from:  Energy: {val_loss_from[0]:.6f} \t Zenith: {val_loss_from[1]:.6f} \t Azimuth {val_loss_from[2]:.6f}, \t Hist {val_loss_from[3]:.6f}")
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
                        return current_epoch

                    if current_epoch != epochs:
                        pbar          = tqdm(total = steps_per_epoch*n_steps, position=0, leave = True)

                    learning_rate = next(lr_schedule)
                    opt.learning_rate.assign(learning_rate)

                    time_avg=tot_time/current_epoch
                    if current_epoch % val_epoch == 0:
                        model.save(save_path)
                        print("Model saved")
                        if wandblog:
                            loader_test = DisjointLoader(dataset_test, batch_size=batch_size, epochs=1)
                            fig, _ = performance_plot(loader_test, test_step, metrics, bins=20, save=True, save_path=save_path)
                            title="performanceplot_"+str(current_epoch)
                            wandb.log({title: [wandb.Image(fig, caption=title)]})
                
                    loss            = 0
                    start_time      = time.time()
                    current_epoch  += 1
                    current_batch   = 0
    return current_epoch
    run.finish()
    
######
#dependencies
######
def get_lr_schedule(construct_dict):
    schedule  = construct_dict['run_params']['lr_schedule']
    lr_0        = construct_dict['run_params']['learning_rate']
    warm_up        = construct_dict['run_params']['warm_up']
    decay       = construct_dict['run_params']['lr_decay']

    import dev.lr_schedules as lr_module

    lr_generator = getattr(lr_module, schedule)

    lr_schedule  = lr_generator(lr_0, warm_up, decay)()

    return lr_schedule



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

def setup_model(construct_dict):
    # Retrieve name and params for construction
    model_name    = construct_dict['model_name']
    hyper_params  = construct_dict['hyper_params']
    experiment    = construct_dict['experiment']

    # Load model from model folder
    import dev.models as models
    model         = getattr(models, model_name) 
    model         = model(**hyper_params)

    # Make folder for saved states
    model_path    = osp.join(cwd, "trained_models", experiment)
    if not osp.isdir(model_path):
        os.makedirs(model_path)

    return model, model_path

