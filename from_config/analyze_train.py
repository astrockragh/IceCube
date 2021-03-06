import os, sys, tqdm, json, shutil, glob

import os.path as osp

from tensorflow.keras.backend import clear_session
import tensorflow as tf

gpu_devices = tf.config.list_physical_devices('GPU') 
if len(gpu_devices) > 0:
    print("GPU detected")
    for i in range(len(gpu_devices)):
        tf.config.experimental.set_memory_growth(gpu_devices[i], True)

###find new options###

# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5) #define how much vram to use
# sess=tf.compat.v1.Session(config=(tf.ConfigProto(gpu_options=gpu_options)))

exp0_folder = str(sys.argv[1])
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
if len(sys.argv)==3:
    os.environ['TF_NUM_INTRAOP_THREADS'] = str(sys.argv[2])
SHUTDOWN = False
##########################################################
#      Loop over JSON files and train models             # 
##########################################################

# Generate list over experiments to run
from dev.utils import list_experiments, clean_done, check_dataset
from dev.analyze_script import analyze_train
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

    # data_exists=check_dataset(construct_dict['data_params']['database'], construct_dict['data_params']['muon'],\
    #      construct_dict['data_params']['n_data'], construct_dict['data_params']['graph_construction'])
    # if data_exists:
    #     print('No data construction required')
    #     construct_dict['data_params']['restart']=False


    print(f"Starting experiment from {experiment[:-5]}")

    epochexit=analyze_train(construct_dict)
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





    
    


