import numpy as np
import os, sqlite3, pickle, sys, gzip, shutil
if hasattr(__builtins__,'__IPYTHON__'):
    print('Notebook')
    from tqdm.notebook import tqdm
else:
    print('Not notebook')
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
        targets= ["energy_log10", "zenith","azimuth", "event_no"],\
            transform_path='../db_files/dev_lvl7/transformers.pkl',\
                db_path= '../db_files/dev_lvl7/dev_lvl7_mu_nu_e_classification_v003.db',\
                 set_path='../db_files/dev_lvl7/sets.pkl',\
                  n_neighbors = 30, restart=False, graph_construction='classic', **kwargs):


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
        super().__init__(**kwargs)
    
    @property
    def path(self):
        """
        Set the path of the data to be in the processed folder
        """
        cwd = osp.abspath('')
        path = osp.join(cwd, f"processed/{self.n_steps}_type_{self.graph_construction}_nn_{self.n_neighbors}")
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
        return train_events['event_no'].to_numpy(), test_events['event_no'].to_numpy()
    
    def check_dataset(self):
        return osp.exists(self.path)

    def download(self):
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
            print("Reading files")
            df_truth=read_sql(f"select event_no from truth", conn)
            splits=np.array_split(np.sort(df_truth['event_no'].to_numpy()),self.n_steps)
            start_ids, stop_ids=[],[]
            for i in range(self.n_steps):
                start_ids.append(splits[i][0])
                stop_ids.append(splits[i][-1])
                
            train_events, test_events=self.get_event_no()
            df_sort=df_truth.sort_values('event_no')
            df_sort=df_sort.reset_index(drop=True)
            df_test=df_sort[df_sort['event_no'].isin(test_events)]
            df_train=df_sort[df_sort['event_no'].isin(train_events)]
            testid, trainid=df_test.index.to_numpy(), df_train.index.to_numpy()
            
            pickle.dump(testid, open(osp.join(self.path, "testid.pkl"), 'wb'))
            pickle.dump(trainid, open(osp.join(self.path, "trainid.pkl"), 'wb'))
            print('Starting loop')
            for i, (start_id, stop_id) in enumerate(zip(start_ids, stop_ids)):
                df_event = read_sql(f"select event_no from features where event_no >= {start_id} and event_no <= {stop_id}", conn)
                print('Events read')
                df_feat  = read_sql(f"select {feature_call} from features where event_no >= {start_id} and event_no <= {stop_id}", conn)
                print('Features read')
                df_targ  = read_sql(f"select {target_call} from truth    where event_no >= {start_id} and event_no <= {stop_id}", conn)
                print('Targets read, transforming')
                transformers = pickle.load(open(self.transform_path, 'rb'))
                trans_x      = transformers['features']
                trans_y      = transformers['truth']


                for col in ["dom_x", "dom_y", "dom_z"]:
                    df_feat[col] = trans_x[col].inverse_transform(np.array(df_feat[col]).reshape(1, -1)).T/self.dom_norm

                for col in ["energy_log10", "zenith","azimuth"]:
                    # print(col)
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

                graph_list=[]
                # Generate adjacency matrices
                for x, y in tqdm(zip(xs, ys), total = len(xs)):
                    try:
                        a = knn(x[:, :3], self.n_neighbors)
                    except:
                        a = csr_matrix(np.ones(shape = (x.shape[0], x.shape[0])) - np.eye(x.shape[0]))


                    graph_list.append(Graph(x = x, a = a, y = y))
                print('List->array')
                graph_list = np.array(graph_list, dtype = object)

                print("Saving dataset")
                pickle.dump(graph_list, open(osp.join(self.path, f"data_{i}.dat"), 'wb'))

            
    def read(self):
        if self.restart and self.k==0:
            self.reload()
            self.download()
            self.k+=1
        print("Loading data to memory")
        data=[]
        for i in tqdm(range(self.n_steps)):
            datai  = pickle.load(open(osp.join(self.path, f"data_{i}.dat"), 'rb'))
            for graph in datai:
                data.append(graph)
        
        self.train_idx=pickle.load(open(osp.join(self.path, "trainid.pkl"), 'rb'))
        self.test_idx=pickle.load(open(osp.join(self.path, "testid.pkl"), 'rb'))
      
        
        return data