import numpy as np
import os, sqlite3, pickle, sys, gzip, shutil, time
from tqdm import tqdm
import os.path as osp

from pandas import read_sql, read_pickle, concat, read_csv, DataFrame
from sklearn.preprocessing import normalize, RobustScaler
from sklearn.neighbors import kneighbors_graph as knn
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from spektral.data import Dataset, Graph
from scipy.sparse import csr_matrix

class graph_data(Dataset):
    """
    data that takes config file
    """

    def __init__(self, n_steps=10 ,features=["dom_x", "dom_y", "dom_z", "dom_time", "charge_log10", "width", "rqe"], \
        targets= ["energy_log10", "zenith","azimuth"],\
            transform_path='../db_files/dev_lvl7/transformers.pkl',\
                db_path= '../db_files/dev_lvl7/dev_lvl7_mu_nu_e_classification_v003.db',\
                 set_path='../db_files/dev_lvl7/sets.pkl',\
                  n_neighbors = 30, restart=False, graph_construction='classic', traintest='train', i_train=0, i_test=0, **kwargs):

        self.traintest=traintest
        self.n_steps=n_steps
        self.set_path=set_path
        self.features=features
        self.targets=targets
        self.dom_norm = 1e3
        self.transform_path=transform_path
        self.db_path=db_path
        self.n_neighbors = n_neighbors
        self.restart=restart
        self.graph_construction=graph_construction
        self.k=0
        self.i_test=i_test
        self.i_train=i_train
        super().__init__(**kwargs)
    
    @property
    def path(self):
        """
        Set the path of the data to be in the processed folder
        """
        cwd = osp.abspath('')
        path = osp.join(cwd, f"processed/where_{self.graph_construction}_{self.n_neighbors}")
        return path

    def reload(self):
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
            print('Removed and ready to reload')
    
    def get_event_no(self):
        print('Reading sets')
        sets = read_pickle(self.set_path)
        train_events = sets['train']
        test_events = sets['test']
        if self.graph_construction=='classic':
            return train_events['event_no'].to_numpy(), test_events['event_no'].to_numpy()
        if self.graph_construction=='muon':
            return train_events.to_numpy(), test_events.to_numpy()
    def check_dataset(self):
        return osp.exists(self.path)

    def download(self):
        self.k+=1
        # Get raw_data
        db_file   = self.db_path

        # Make output folder
        os.makedirs(self.path)

        print("Connecting to db-file")
        with sqlite3.connect(db_file) as conn:
            # Find indices to cut after

            # SQL queries format
            feature_call = ", ".join(self.features)
            target_call  = ", ".join(self.targets)

            # Load data from db-file

            train_events, test_events=self.get_event_no()
            np.random.shuffle(train_events)
            np.random.shuffle(test_events)
            train_events = np.array_split(train_events,self.n_steps)
            test_events  = np.array_split(test_events,self.n_steps)

            for i, (train, test) in enumerate(zip(train_events, test_events)):
                for tt, events in zip(['train', 'test'], [train, test]):
                    print("Reading files")
                    start=time.time()
                    df_feat  = read_sql(f"select {feature_call}, event_no from features where event_no in {tuple(events)}", conn).sort_values('event_no')
                    print('Features read')
                    df_targ  = read_sql(f"select {target_call}, event_no from truth where event_no in {tuple(events)}", conn).sort_values('event_no')
                    print('Targets read, transforming')
                    transformers = pickle.load(open(self.transform_path, 'rb'))
                    stop=time.time()
                    print(f"Reading {tt} {i} took {np.round(stop-start, 2)} s")
                    trans_x      = transformers['features']
                    trans_y      = transformers['truth']
                    for col in ["dom_x", "dom_y", "dom_z"]:
                        df_feat[col] = trans_x[col].inverse_transform(np.array(df_feat[col]).reshape(1, -1)).T/self.dom_norm

                    for col in ["energy_log10", "zenith","azimuth"]:
                        # print(col)
                        df_targ[col] = trans_y[col].inverse_transform(np.array(df_targ[col]).reshape(1, -1)).T



                    # Cut indices
                    print("Splitting data to events")
                    idx_list    = np.array(df_feat['event_no'])
                    df_feat.drop('event_no', axis=1, inplace=True)
                    x_not_split = np.array(df_feat)

                    _, idx = np.unique(idx_list.flatten(), return_index = True) 
                    xs          = np.split(x_not_split, idx[1:])

                    ys          = np.array(df_targ)
                    print(df_feat.head())
                    print(df_targ.head())

                    graph_list=[]
                    # Generate adjacency matrices
                    for x, y in tqdm(zip(xs, ys), total = len(xs), position=1, desc=f'Transform {tt} {i}'):
                        try:
                            a = knn(x[:, :3], self.n_neighbors)
                        except:
                            a = csr_matrix(np.ones(shape = (x.shape[0], x.shape[0])) - np.eye(x.shape[0]))
                        graph_list.append(Graph(x = x, a = a, y = y))
                    print('List->array')
                    graph_list = np.array(graph_list, dtype = object)
                
                    print(f"Saving dataset {tt} {i}: {len(graph_list)} {tt}")
                    pickle.dump(graph_list, open(osp.join(self.path, f"{tt}_{i}.dat"), 'wb'))
                    stop=time.time()
                    print(f"Process {tt} {i} took {np.round(stop-start, 2)} s")
                    
 
    def read(self):
        if self.restart and self.k==0:
            self.reload()
            self.download()
        
        
        if self.traintest=='train':
            print(f"Loading train data {self.i_train} to memory")
            data  = pickle.load(open(osp.join(self.path, f"train_{self.i_train}.dat"), 'rb'))

        if self.traintest=='test':
            print(f"Loading test data {self.i_test} to memory")
            data  = pickle.load(open(osp.join(self.path, f"test_{self.i_test}.dat"), 'rb'))
        return data