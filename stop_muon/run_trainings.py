import os, sys, tqdm, json, shutil

import os.path as osp

from tensorflow.keras.backend import clear_session


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

gpu_devices = tf.config.list_physical_devices('GPU') 
if len(gpu_devices) > 0:
    print("GPU detected")
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)


SHUTDOWN = False
##########################################################
#      Loop over JSON files and train models             # 
##########################################################

# Generate list over experiments to run
from dev.utils import list_experiments, check_dataset
from dev.train_script import train_model
exp_folder, exp_list = list_experiments()

print(f"Starting process with {len(exp_list)} experiments")
print(exp_list)
# Loop over the experiments
for i, experiment in enumerate(exp_list):

    # Load construction dictionairy from json file
    with open(osp.join(exp_folder, experiment)) as file:
        construct_dict = json.load(file)
    
    data_exists=check_dataset(construct_dict['data_params']['database'], construct_dict['data_params']['muon'],\
         construct_dict['data_params']['n_data'], construct_dict['data_params']['graph_construction'])
    if data_exists:
        print('No data construction required')
        construct_dict['data_params']['restart']=False


    print(f"Starting experiment from {experiment[:-5]}")

    epochexit=train_model(construct_dict)
    print(f'Exited training after {epochexit} epochs')
    shutil.move(osp.join(exp_folder, experiment), "experiments/done")
    print(f"Experiment {experiment[:-5]} done \t {experiment}: {i + 1} / {len(exp_list)}")




    clear_session()

# if SHUTDOWN == True:
#     os.system("shutdown -h")

    # Create a script to go through and test the performance
    # test_model(model = construct_dict['Experiment'], data = instructions_to_dataset_name(construct_dict))
    




# We can setup a shutdown maybe
#os.system("shutdown -h 5")





    
    


