import os, sys, tqdm, json, shutil, glob

import os.path as osp

from tensorflow.keras.backend import clear_session

import tensorflow as tf

# gpu_devices = tf.config.list_physical_devices('GPU') 
# if len(gpu_devices) > 0:
#     print("GPU detected")
#     tf.config.experimental.set_memory_growth(gpu_devices[0], True)

exp_folder = str(sys.argv[1])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
##########################################################
#      Loop over JSON files and train models             # 
##########################################################

# Generate list over experiments to run
from dev.utils import list_experiments, clean_done, check_dataset
from dev.train_script import train_model
clean_done(exp_folder)
exp_folder, exp_list = list_experiments(exp_folder)

print(f"Starting process with {len(exp_list)} experiments")
print(exp_list)
# Loop over the experiments
for i, experiment in enumerate(exp_list):

    # Load construction dictionary from json file
    with open(osp.join(exp_folder, experiment)) as file:
        construct_dict = json.load(file)
    construct_dict['experiment_name']=experiment[:-5]

    construct_dict['data_params']['n_data']=1e3
    construct_dict['experiment']='test'
    construct_dict['run_params']['epochs']=5
    construct_dict['run_params']['val_epoch']=2

    print(f"Starting experiment from {experiment[:-5]}")
    epochexit=train_model(construct_dict)

    clear_session()

# if SHUTDOWN == True:
#     os.system("shutdown -h")

    # Create a script to go through and test the performance
    # test_model(model = construct_dict['Experiment'], data = instructions_to_dataset_name(construct_dict))
    




# We can setup a shutdown maybe
#os.system("shutdown -h 5")





    
    


