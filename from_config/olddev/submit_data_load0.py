import numpy as np
import os, sqlite3, pickle, sys, gzip, shutil
if hasattr(__builtins__,'__IPYTHON__'):
    print('Notebook')
    from tqdm.notebook import tqdm
else:
    print('Not notebook')
    from tqdm import tqdm
import os.path as osp

from pandas import read_sql, concat
import pandas as pd
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

    def __init__(self, frac_data = 0.5 ,features=["dom_x", "dom_y", "dom_z", "dom_time", "charge_log10", "width", "rqe"], 
    targets= ["energy_log10", "zenith","azimuth"],
    transform_path='../../../../pcs557/databases/dev_lvl7_mu_nu_e_classification_v003/meta/transformers.pkl',
    db_path= '../../../../pcs557/databases/dev_lvl7_mu_nu_e_classification_v003/data/dev_lvl7_mu_nu_e_classification_v003.db', 
    set_path='../../../../pcs557/databases/dev_lvl7_mu_nu_e_classification_v003/meta/sets.pkl',
    n_neighbors = 30, restart=False, graph_construction='classic', database='submit', **kwargs):


        self.frac_data = frac_data
        self.features=features
        self.targets=targets
        self.dom_norm = 1e3
        self.transform_path=transform_path
        self.db_path=db_path
        self.set_path=set_path
        self.n_neighbors = n_neighbors
        self.seed = 42
        self.restart=restart
        self.graph_construction=graph_construction
        self.database=database
        super().__init__(**kwargs)
    
    @property
    def path(self):
        """
        Set the path of the data to be in the processed folder
        """
        cwd = osp.abspath('')
        path = osp.join(cwd, f"processed/{self.database}_{self.n_neighbors}nn_{self.graph_construction}graph_{len(self.features)}feat")
        return path

    def reload(self):
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
            print('Removed and ready to reload')

    def get_event_no(self):
        print('Reading sets')
        sets = pd.read_pickle(self.set_path)
        train_events = sets['train']
        test_events = sets['test']
        return train_events['event_no'].to_numpy(), test_events['event_no'].to_numpy()

    def download(self):
        # Get raw_data
        # Make output folder
        os.makedirs(self.path)
        train_events, test_events=self.get_event_no()
        events=[train_events,test_events]
        for i, train_test in enumerate(['train','test']):
            db_file  = self.db_path
            print("Connecting to db-file")
            with sqlite3.connect(db_file) as conn:
                # SQL queries format
                feature_call = ", ".join(self.features)
                target_call  = ", ".join(self.targets)
                event_nos=tuple(events[i].reshape(1, -1)[0])
                # Load data from db-file
                print("Reading files")
                df_event = read_sql(f"select event_no from features where event_no in {event_nos}", conn)
                print("Events read")
                df_feat  = read_sql(f"select {feature_call} from features where event_no in {event_nos}", conn)
                print("Features read")
                df_targ  = read_sql(f"select {target_call } from truth where event_no in {event_nos}", conn)
                print("Truth read")

                transformers = pickle.load(open(self.transform_path, 'rb'))
                trans_x      = transformers['features']
                trans_y      = transformers['truth']


                for col in ["dom_x", "dom_y", "dom_z"]:
                    df_feat[col] = trans_x[col].inverse_transform(np.array(df_feat[col]).reshape(1, -1)).T/self.dom_norm

                for col in df_targ.columns:
                    df_targ[col] = trans_y[col].inverse_transform(np.array(df_targ[col]).reshape(1, -1)).T
            
            

                # Cut indices
                print("Splitting data to events")
                idx_list    = np.array(df_event)
                x_not_split = np.array(df_feat)

                _, idx, counts = np.unique(idx_list.flatten(), return_index = True, return_counts = True) 
                xs          = np.split(x_not_split, np.cumsum(counts)[:-1])

                ys          = np.array(df_targ)
                print(df_feat.head())
                print(df_targ.head())

                # Generate adjacency matrices
                print("Generating adjacency matrices")
                graph_list = []
                for x, y in tqdm(zip(xs, ys), total = len(xs)):
                    try:
                        a = knn(x[:, :3], self.n_neighbors)
                    except:
                        a = csr_matrix(np.ones(shape = (x.shape[0], x.shape[0])) - np.eye(x.shape[0]))


                    graph_list.append(Graph(x = x, a = a, y = y))

                graph_list = np.array(graph_list, dtype = object)


                print("Saving dataset")
                pickle.dump(graph_list, open(osp.join(self.path, f"data_{train_test}.dat"), 'wb'))
        
    def read(self):
        if self.restart:
            self.reload()
            self.download()
        print("Loading train data to memory")
        data_train   = pickle.load(open(osp.join(self.path, "data_train.dat"), 'rb'))
        print("Loading test data to memory")
        data_test   = pickle.load(open(osp.join(self.path, "data_test.dat"), 'rb'))
        
        dataset_train = data_train[:int(self.frac_data*len(data_train))]

        dataset_test  = data_test[:int(self.frac_data*len(data_test))]

        return dataset_train, dataset_test