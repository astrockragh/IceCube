import os, sys, tqdm, json, shutil, glob, argparse

import os.path as osp

from tensorflow.keras.backend import clear_session

import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
parser = argparse.ArgumentParser()
parser.add_argument("-gpu", "--gpu", type=str, required=False)
parser.add_argument("-cpus", "--cpus", type=str, required=False)
args = parser.parse_args()

if args.gpu!='all':
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
gpu_devices = tf.config.list_physical_devices('GPU') 

if len(gpu_devices) > 0:
    print("GPU detected")
    for i in range(len(gpu_devices)):
        tf.config.experimental.set_memory_growth(gpu_devices[i], True)

if args.cpus!='all':
    os.environ['TF_NUM_INTRAOP_THREADS'] = args.cpus
##########################################################
#      Loop over JSON files and train models             # 
##########################################################

# Generate list over experiments to run
from dev.utils import perms
from dev.train_script import train_model

# Load construction dictionary from json file
with open("exp_compare/diff.json") as file:
    diff = json.load(file)
    
keys=list(diff.keys())
exp_list=perms(diff)
# Load construction dictionary from json file
with open("exp_compare/base.json") as file:
    base = json.load(file)

print(f"Starting process with {len(exp_list)} experiments")
print(diff)
# Loop over the experiments

if len(exp_list)>100:
    print('Selected too many compare run, running 100 random runs')
    idxs = np.random.permutation(len(exp_list))
    idxs[:100]
    exp_list=exp_list[idxs]

for i in range(len(exp_list)):
    construct_dict=base
    print('Exploring', keys)
    print('Currently doing', exp_list[i])
    # if i==0:
    #     construct_dict['data_params']['restart']=True
    # else:
    #     construct_dict['data_params']['restart']=False
    for j, key in enumerate(keys):
        if key in construct_dict['run_params']:
            typ=type(construct_dict['run_params'][key])
            construct_dict['run_params'][key]=typ(exp_list[i][j])
        elif key in construct_dict['hyper_params']:
            typ=type(construct_dict['hyper_params'][key])
            construct_dict['hyper_params'][key]=typ(exp_list[i][j])
        elif key in construct_dict['data_params']:
            typ=type(construct_dict['data_params'][key])
            construct_dict['data_params'][key]=typ(exp_list[i][j])
    #make_title
    title=''
    for key, val in zip(keys, exp_list[i]):
        title+=key[:2]+str(val)
    construct_dict['experiment_name']=title
    epochexit=train_model(construct_dict)
    print(f'Exited training after {epochexit} epochs')
    print(f"Experiment {i} done: {i + 1} / {len(exp_list)}")




    clear_session()

# if SHUTDOWN == True:
#     os.system("shutdown -h")

    # Create a script to go through and test the performance
    # test_model(model = construct_dict['Experiment'], data = instructions_to_dataset_name(construct_dict))
    




# We can setup a shutdown maybe
#os.system("shutdown -h 5")





    
    


