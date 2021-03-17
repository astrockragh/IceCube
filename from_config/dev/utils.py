import os, sys, json, shutil
import os.path as osp
import numpy as np

from importlib import import_module


cwd = osp.abspath('')


def list_experiments():
    experiment_folder = osp.join(cwd, "experiments/todo") 
    experiment_files  = os.listdir(experiment_folder)
    return experiment_folder, experiment_files

def check_dataset(database='MuonGun', muon=True, n_data=1e4, graph_construction='classic'):
    """
    Check if a given dataset is generated, else initiate the process
    Return data_exists, as_exists
    Boolean determing if x data file and as data file are constructed
    """
    n_data=int(n_data)
    data_folder = osp.join(cwd, f"processed/{database}_muon_{muon}_n_data_{n_data}_type_{graph_construction}")
    return osp.exists(data_folder)




