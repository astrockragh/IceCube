import numpy as np
import os, sqlite3, pickle, sys, gzip, shutil
if hasattr(__builtins__,'__IPYTHON__'):
    print('Notebook')
    from tqdm.notebook import tqdm
else:
    print('Not notebook')
    from tqdm import tqdm
import os.path as osp

from pandas import read_sql, concat, read_csv, DataFrame
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

    def __init__(self, n_data = 1e6 ,features=["dom_x", "dom_y", "dom_z", "dom_time", "charge_log10", "rqe"], \
        targets= ["energy_log10", "zenith","azimuth"], pid = None,\
    transform_path="../../../../pcs557/databases/dev_lvl7_mu_nu_e_classification_v003/meta/transformers.pkl",\
    db_path="../../../../pcs557/databases/dev_lvl7_mu_nu_e_classification_v003/data/dev_lvl7_mu_nu_e_classification_v003.db",\
                  n_neighbors = 10, restart=False, data_split = [0.8, 0.1, 0.1], graph_construction='classic', database='dev7_v003', **kwargs):


        self.n_data = int(n_data)
        self.features=features
        self.targets=targets
        self.pid=pid
        self.dom_norm = 1e3
        self.transform_path=transform_path
        self.db_path=db_path
        self.n_neighbors = n_neighbors
        if sum(data_split) != 1:
            sys.exit("Total splits must add up to 1")
        self.train_size, self.val_size, self.test_split = data_split
        self.seed = 42
        self.restart=restart
        self.graph_construction=graph_construction
        self.database=database
        self.k=0
        super().__init__(**kwargs)
    
    @property
    def path(self):
        """
        Set the path of the data to be in the processed folder
        """
        cwd = osp.abspath('')
        path = osp.join(cwd, f"processed/{self.database}_pid_{self.pid}_n_data_{self.n_data}_type_{self.graph_construction}")
        return path

    def reload(self):
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
            print('Removed and ready to reload')

    def check_dataset(self):
        return osp.exists(self.path)

    def download(self):
        if not self.check_dataset():
            # Get raw_data
            db_file   = self.db_path

            # Make output folder
            os.makedirs(self.path)

            print("Connecting to db-file")
            with sqlite3.connect(db_file) as conn:
                print('Loading dataframes')
                target_call  = ", ".join(self.targets)
                df_targ=read_sql(f"select {target_call} from truth limit {self.n_data}", conn)
                pid=['event_no', 'pid']
                pid_call = ", ".join(pid)
                df_pid=read_sql(f"select {pid_call} from truth limit {self.n_data}", conn)
                ##what particles?
                if self.pid==None:
                    event_nos=np.array(df_pid['event_no'])
                if self.pid!=None and len(self.pid==1): 
                    event_nos=np.array(df_pid['event_no'][df_pid['pid']!=self.pid[0]])
                if self.pid!=None and len(self.pid>1): 
                    event_nos=np.array(df_pid['event_no'][df_pid['pid'].isin(self.pid)])
                # SQL queries format
                feature_call = ", ".join(self.features)
                
                # Load data from db-file
                print("Reading files")
    
                str_eventnos=[str(event_no) for event_no in event_nos]
                event_nocall=", ".join(str_eventnos)
                df_feat=read_sql(f"select event_no, {feature_call} from features where event_no in ({event_nocall})", conn)
                
                df_event=DataFrame(df_feat['event_no'])
                df_feat.drop(['event_no'], axis=1)
                transformers = pickle.load(open(self.transform_path, 'rb'))
                trans_x      = transformers['features']
                trans_y      = transformers['truth']
                

                for col in ["dom_x", "dom_y", "dom_z"]:
                    df_feat[col] = trans_x[col].inverse_transform(np.array(df_feat[col]).reshape(1, -1)).T/self.dom_norm

                for col in df_targ.columns:
                    # print(col)
                    df_targ[col] = trans_y[col].inverse_transform(np.array(df_targ[col]).reshape(1, -1)).T
            
            

                # Cut indices
                print("Splitting data to events")
                idx_list    = np.array(df_feat)
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
                pickle.dump(graph_list, open(osp.join(self.path, "data.dat"), 'wb'))
                df_csv=DataFrame(df_pid['event_no'])
                df_csv.to_csv(self.path+'/event_nos.csv')
        else:
            pass
        
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
        
        df_event=read_csv(self.path+'/event_nos.csv')
        self.df_event=df_event
        
        return data