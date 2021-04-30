import os, sys, json, shutil
import os.path as osp
import numpy as np

from importlib import import_module


cwd = osp.abspath('')
def to_sigma(zes, azs):
    from scipy.special import iv
    az_sigma = np.sqrt(1 - iv(1,np.square(azs))/iv(0,np.square(azs)))*180/np.pi
    ze_sigma = np.sqrt(1 - iv(1,np.square(zes))/iv(0,np.square(zes)))*180/np.pi
    return ze_sigma, az_sigma

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

def perms(diffs):
    from itertools import product
    keys=list(diffs.keys())
    val=list(diffs.values())
    for i, s in enumerate(val):
        if i==0:
            a=val[0]
        else:
            a=product(a, val[i])
    bs=[]
    for b in a:
        bs.append(b)
    output=[]
    def removeNestings(l):
        for i in l:
            if type(i) == tuple:
                removeNestings(i)
            else:
                output.append(i)
    removeNestings(bs)
    perms=np.array(output)
    return perms.reshape(-1, len(keys))


