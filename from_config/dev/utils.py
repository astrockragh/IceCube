import os, sys, json, shutil
import os.path as osp
import numpy as np

from importlib import import_module


cwd = osp.abspath('')


def list_experiments(folder):
    experiment_folder = osp.join(cwd, folder, "todo") 
    experiment_files  = os.listdir(experiment_folder)
    return experiment_folder, experiment_files

def clean_done(folder):
    experiment_folder = osp.join(cwd, folder, "done") 
    legacy_path=osp.join(cwd, folder, 'legacy')
    if not osp.exists(legacy_path):
        os.mkdir(legacy_path)
        print('Legacy made')
    try:
        files  = os.listdir(experiment_folder)
        for f in files:
            shutil.move(osp.join(experiment_folder,f), folder, 'legacy/')
    except:
        os.mkdir(experiment_folder)
    print('Cleaned done folder')

def check_dataset(database='MuonGun', muon=True, n_data=1e4, graph_construction='classic'):
    """
    Check if a given dataset is generated, else initiate the process
    Return data_exists, as_exists
    Boolean determing if x data file and as data file are constructed
    """
    n_data=int(n_data)
    data = osp.join(cwd, f"processed/{database}_muon_{muon}_n_data_{n_data}_type_{graph_construction}/data.dat")
    return osp.exists(data)




