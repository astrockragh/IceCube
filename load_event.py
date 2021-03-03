import numpy as np
import os, sqlite3, pickle, sys, gzip, shutil, time
import os.path as osp

from pandas import read_sql, concat
from sklearn.preprocessing import normalize, RobustScaler
from sklearn.neighbors import kneighbors_graph as knn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from spektral.data import Dataset, Graph
import tensorflow as tf

features = ["dom_x", "dom_y", "dom_z", "time", "charge_log10"]
targets  = ["energy_log10", "zenith","azimuth"]

class load_event(Dataset):
    """
    First hopefully working version of the data
    """

    def __init__(self, event=0, event_no=None, transform=True, muon = True, n_neighbors = 6, restart=True, **kwargs):
        self.skip   = event
        self.event_no = event_no
        self.n_neighbors = n_neighbors
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
        feature_call = ", ".join(features)
        target_call  = ", ".join(targets)
        with sqlite3.connect(db_file) as conn:
            if self.event_no is None:
                # Find indices to cut after
                try:
                    if self.muon:
                        print('Loading Muons')
                        start_id = conn.execute(f"select distinct event_no from features where event_no > 138674340 limit 1 offset {self.skip}").fetchall()[0][0]
                        stop_id  = conn.execute(f"select distinct event_no from features where event_no > 138674340 limit 1 offset {self.skip + 1}").fetchall()[0][0]
                    else:
                        print('Loading Neutrinos')
                        start_id = conn.execute(f"select distinct event_no from features limit 1 offset {self.skip}").fetchall()[0][0]
                        stop_id  = conn.execute(f"select distinct event_no from features limit 1 offset {self.skip + 1}").fetchall()[0][0]
                except:
                    ""
                    start_id = 0
                    stop_id  = 999999999

                # Load data from db-file
                print("Reading files")
                self.index=start_id
                df_event = read_sql(f"select event_no       from features where event_no >= {start_id} and event_no < {stop_id}", conn)
                df_feat  = read_sql(f"select {feature_call} from features where event_no >= {start_id} and event_no < {stop_id}", conn)
                df_targ  = read_sql(f"select {target_call } from truth    where event_no >= {start_id} and event_no < {stop_id}", conn)
            else:
                # Load data from db-file
                print("Reading files")
                df_event = read_sql(f"select event_no       from features where event_no == {self.event_no}", conn)
                df_feat  = read_sql(f"select {feature_call} from features where event_no == {self.event_no}", conn)
                df_targ  = read_sql(f"select {target_call } from truth    where event_no == {self.event_no}", conn)
            
            if self.transform:
                transformers = pickle.load(open(osp.join(db_folder, "transformers.pkl"), 'rb'))
                trans_x      = transformers['features']
                trans_y      = transformers['truth']


                for col in df_feat.columns:
                    df_feat[col] = trans_x[col].inverse_transform(np.array(df_feat[col]).reshape(1, -1)).T

                for col in df_targ.columns:
                    df_targ[col] = trans_y[col].inverse_transform(np.array(df_targ[col]).reshape(1, -1)).T

            idx_list    = np.array(df_event)
            x_not_split = np.array(df_feat)

            _, idx, counts = np.unique(idx_list.flatten(), return_index = True, return_counts = True) 
            xs          = np.split(x_not_split, np.cumsum(counts)[:-1])

            ys          = np.array(df_targ)
            # Generate adjacency matrices
            print("Generating adjacency matrices")
            graph_list = []
            for x, y in zip(xs, ys):
                a = knn(x[:, :3], self.n_neighbors)

                graph_list.append(Graph(x = x, a = a, y = y))

            graph_list = np.array(graph_list, dtype = object)


            print("Saving dataset")
            pickle.dump(graph_list, open(osp.join(self.path, "inspect_event.dat"), 'wb'))

        
    def read(self):
        if self.restart and self.k==0:
            self.reload()
            self.download()
            self.k+=1
        print("Loading data to memory")
        data   = pickle.load(open(osp.join(self.path, "inspect_event.dat"), 'rb'))

        if self.event_no is not None:
            self.index = self.event_no

        return data