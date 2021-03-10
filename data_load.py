import numpy as np
import os, sqlite3, pickle, sys, gzip, shutil, time
from tqdm.notebook import tqdm
import os.path as osp

from pandas import read_sql, concat
from sklearn.preprocessing import normalize, RobustScaler
from sklearn.neighbors import kneighbors_graph as knn
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from spektral.data import Dataset, Graph
import tensorflow as tf

features = ["dom_x", "dom_y", "dom_z", "time", "charge_log10"]
# targets  = ["energy_log10", "zenith","azimuth"]

targets  = ["energy_log10", "position_x", "position_y", "position_z", "direction_x", "direction_y", "direction_z"]

class graph_data(Dataset):
    """
    First hopefully working version of the data
    """

    def __init__(self, n_data = None, transform=True, muon = True, skip = 0, n_neighbors = 6, restart=False, train_val_test_split = [0.8, 0.1, 0.1], **kwargs):
        self.n_data = n_data
        self.skip   = skip
        self.n_neighbors = n_neighbors
        if sum(train_val_test_split) != 1:
            sys.exit("Total splits must add up to 1")
        self.train_size, self.val_size, self.test_split = train_val_test_split
        self.seed = 42
        self.transform=transform
        self.restart=restart
        self.muon=muon
        self.k=0
        super().__init__(**kwargs)

    @property
    def path(self):
        """
        Set the path of the data to be in the processed folder
        """
        cwd = osp.abspath('')
        path = osp.join(cwd, "processed/graph_dataset")
        return path

    def reload(self):
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
            print('Removed and ready to reload')

    def download(self):
        download_start = time.time()
        # Get raw_data
        db_folder = osp.join(osp.abspath(''), "db_files")
        db_file   = osp.join(db_folder, "rasmus_classification_muon_3neutrino_3mio.db")

        # Make output folder
        os.makedirs(self.path)

        print("Connecting to db-file")
        with sqlite3.connect(db_file) as conn:
            # Find indices to cut after
            try:
                if self.muon:
                    print('Loading Muons')
                    start_id = conn.execute(f"select distinct event_no from features where event_no>=138674340 limit 1 offset {self.skip}").fetchall()[0][0]
                    stop_id  = conn.execute(f"select distinct event_no from features where event_no>=138674340 limit 1 offset {self.skip + self.n_data}").fetchall()[0][0]
                else:
                    print('Loading Neutrinos')
                    start_id = conn.execute(f"select distinct event_no from features limit 1 offset {self.skip}").fetchall()[0][0]
                    stop_id  = conn.execute(f"select distinct event_no from features limit 1 offset {self.skip + self.n_data}").fetchall()[0][0]
            except:
                ""
                start_id = 0
                stop_id  = 999999999

            # SQL queries format
            feature_call = ", ".join(features)
            target_call  = ", ".join(targets)

            # Load data from db-file
            print("Reading files")
            df_event = read_sql(f"select event_no       from features where event_no >= {start_id} and event_no < {stop_id}", conn)
            df_feat  = read_sql(f"select {feature_call} from features where event_no >= {start_id} and event_no < {stop_id}", conn)
            df_targ  = read_sql(f"select {target_call } from truth    where event_no >= {start_id} and event_no < {stop_id}", conn)
            
            if self.transform:
                transformers = pickle.load(open(osp.join(db_folder, "transformers.pkl"), 'rb'))
                trans_x      = transformers['features']
                trans_y      = transformers['truth']


                for col in ["dom_x", "dom_y", "dom_z"]:
                    df_feat[col] = trans_x[col].inverse_transform(np.array(df_feat[col]).reshape(1, -1)).T

                for col in df_targ.columns:
                    # print(col)
                    df_targ[col] = trans_y[col].inverse_transform(np.array(df_targ[col]).reshape(1, -1)).T
            
            

            # Cut indices
            print("Splitting data to events")
            idx_list    = np.array(df_event)
            x_not_split = np.array(df_feat)

            _, idx, counts = np.unique(idx_list.flatten(), return_index = True, return_counts = True) 
            xs          = np.split(x_not_split, np.cumsum(counts)[:-1])

            ys          = np.array(df_targ)

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
            pickle.dump(graph_list, open(osp.join(self.path, "data.dat"), 'wb'))

        
    def read(self):
        if self.restart and self.k==0:
            self.reload()
            self.download()
            self.k+=1
        print("Loading data to memory")
        data   = pickle.load(open(osp.join(self.path, "data.dat"), 'rb'))


        np.random.seed(self.seed)
        idxs = np.random.permutation(len(data))
        train_split = int(self.train_size * len(data))
        val_split   = int(self.val_size * len(data)) + train_split

        idx_tr, idx_val, idx_test  = np.split(idxs, [train_split, val_split])
        self.index_lists = [idx_tr, idx_val, idx_test]

        return data