import os, sys, tqdm, json, shutil, glob

import os.path as osp

from tensorflow.keras.backend import clear_session

import tensorflow as tf

gpu_devices = tf.config.list_physical_devices('GPU') 
if len(gpu_devices) > 0:
    print("GPU detected")
    for i in range(len(gpu_devices)):
        tf.config.experimental.set_memory_growth(gpu_devices[i], True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
SHUTDOWN = False
##########################################################
#      Loop over JSON files and train models             # 
##########################################################

# Generate list over experiments to run
from dev.utils import list_experiments, clean_done, check_dataset
from dev.train_script import train_model

# Load construction dictionary from json file
with open("experiments/diff.json") as file:
    diff = json.load(file)
keys=list(diff.keys())

# Load construction dictionary from json file
with open("experiments/base.json") as file:
    base = json.load(file)

print(f"Starting process with {len(exp_list)} experiments")
print(exp_list)
# Loop over the experiments
for i in len(keys):

    
    epochexit=train_model(construct_dict)
    print(f'Exited training after {epochexit} epochs')
    shutil.move(osp.join(exp_folder, experiment), osp.join(exp0_folder+"/done", experiment))
    print(f"Experiment {experiment[:-5]} done \t {experiment}: {i + 1} / {len(exp_list)}")




    clear_session()

# if SHUTDOWN == True:
#     os.system("shutdown -h")

    # Create a script to go through and test the performance
    # test_model(model = construct_dict['Experiment'], data = instructions_to_dataset_name(construct_dict))
    




# We can setup a shutdown maybe
#os.system("shutdown -h 5")





    
    


