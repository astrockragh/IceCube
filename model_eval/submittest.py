import os, sys, argparse, importlib,json
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import pandas as pd

gpu_devices = tf.config.list_physical_devices('GPU') 
if len(gpu_devices) > 0:
    print("GPU detected")
    for i in range(len(gpu_devices)):
        tf.config.experimental.set_memory_growth(gpu_devices[i], True)
else:
    print('No GPU detected')


import spektral
from spektral.data import DisjointLoader
os.chdir('../from_config')
import dev.data_load as dl
os.chdir('../model_eval')
##check that prediction name given
save_path='predictions/'+'pred_'+str(sys.argv[1])

graph_data=dl.graph_data
### test this bit
parser = argparse.ArgumentParser()
parser.add_argument("-database", "--database", type=str, required=False)
parser.add_argument("-model", "--model", type=str, required=False)
args = parser.parse_args()


with tf.device('/cpu:0'): # if on the cpu
    model=tf.keras.models.load_model('../from_config/trained_models/IceCube/Sage_sage1nonorm_10_2aauycmh')
    model.compile()
batch_size=512
#just give the same database as you would normally run it on
dataset =graph_data(n_data=100000,skip=0, restart=1, transform=True,\
                    transform_path='db_files/muongun/transformers.pkl',
                    db_path= 'db_files/muongun/rasmus_classification_muon_3neutrino_3mio.db')

#../../../../pcs557/databases/dev_lvl7_mu_nu_e_classification_v003----IC8611_oscNext_003_final/data/meta/transformers.pkl
#../../../../pcs557/databases/dev_lvl7_mu_nu_e_classification_v003---IC8611_oscNext_003_final/data/IC8611_oscNext_003_final.db

## get out relevant stuff
train, val, test=dataset.index_lists
dataset_test=dataset[test]
loader = DisjointLoader(dataset_test, batch_size=batch_size, epochs=1)
df_event=dataset.df_event

### define func

@tf.function(input_signature = loader.tf_signature(), experimental_relax_shapes = True)
def test_step(inputs, targets):
    predictions = model(inputs, training = False)
    targets     = tf.cast(targets, tf.float32) 

    return predictions, targets

## def predict func

def predict(loader):
    prediction_list, target_list = [], []
    for batch in loader:
        inputs, targets = batch
        predictions, targets = test_step(inputs, targets)

        prediction_list.append(predictions.numpy())
        target_list.append(targets.numpy())
        y_reco  = tf.concat(prediction_list, axis = 0).numpy()
        y_true  = tf.concat(target_list, axis = 0)
        y_true  = tf.cast(y_true, tf.float32).numpy()
    return y_reco, y_true

reco, true=predict(loader)

reco_str=['energy_log10_pred', 'zenith_pred', 'azimuth_pred', 'zenith_sigma', 'azimuth_sigma']
recos=pd.DataFrame(reco)
recos.columns=reco_str
recos.head()
recos['event_no']=np.array(df_event['event_no'][test])
recos.head()
recos.to_csv(save_path)
os.system("python submit_results.py -- --save_path {save_path} --model {} --init CKJ")