import os, sys, tqdm, json, shutil, glob, argparse

import os.path as osp

from tensorflow.keras.backend import clear_session
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--f", type=str, required=False)
parser.add_argument("-cpus", "--cpus", type=str, required=False)
args = parser.parse_args()


exp0_folder = str(args.f)
if args.cpus!='all':
    os.environ['TF_NUM_INTRAOP_THREADS'] = args.cpus
SHUTDOWN = False
##########################################################
#      Loop over JSON files and train models             # 
##########################################################

# Generate list over experiments to run
from dev.utils import list_experiments
from dev.generate import test_data
exp_folder, exp_list = list_experiments(exp0_folder)

print(f"Starting process with {len(exp_list)} experiments")
print(exp_list)
# Loop over the experiments
for i, experiment in enumerate(exp_list):
    
    # Load construction dictionary from json file
    with open(osp.join(exp_folder, experiment)) as file:
        construct_dict = json.load(file)
    construct_dict['data_params']['n_steps']=10
    construct_dict['experiment_name']=experiment[:-5]
    construct_dict['data_params']['restart']=True
    construct_dict['wandblog']=False
    # data_exists=check_dataset(construct_dict['data_params']['database'], construct_dict['data_params']['muon'],\
    #      construct_dict['data_params']['n_data'], construct_dict['data_params']['graph_construction'])
    # if data_exists:
    #     print('No data construction required')
    #     construct_dict['data_params']['restart']=False


    print(f"Starting experiment from {experiment[:-5]}")

    test_data(construct_dict)


    clear_session()
