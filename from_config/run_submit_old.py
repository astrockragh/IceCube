import os, sys, tqdm, json, shutil, glob, argparse

import os.path as osp

from tensorflow.keras.backend import clear_session
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--f", type=str, required=False)
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

exp0_folder = str(args.f)
if args.cpus!='all':
    os.environ['TF_NUM_INTRAOP_THREADS'] = args.cpus
SHUTDOWN = False
##########################################################
#      Loop over JSON files and train models             # 
##########################################################

# Generate list over experiments to run
from dev.utils import list_experiments, clean_done, check_dataset
from dev.train_submit_test import train_model
clean_done(exp0_folder)
exp_folder, exp_list = list_experiments(exp0_folder)

print(f"Starting process with {len(exp_list)} experiments")
print(exp_list)
# Loop over the experiments
for i, experiment in enumerate(exp_list):
    
    # Load construction dictionary from json file
    with open(osp.join(exp_folder, experiment)) as file:
        construct_dict = json.load(file)
    construct_dict['experiment_name']=experiment[:-5]
    construct_dict['data_params']['restart']=False



    print(f"Starting experiment from {experiment[:-5]}")

    epochexit=train_model(construct_dict)
    print(f'Exited training after {epochexit} epochs')
    shutil.move(osp.join(exp_folder, experiment), osp.join(exp0_folder+"/done", experiment))
    print(f"Experiment {experiment[:-5]} done \t {experiment}: {i + 1} / {len(exp_list)}")




    clear_session()




    
    


