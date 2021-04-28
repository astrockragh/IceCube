from importlib import reload
import dev.submit_data_load as dl
reload(dl)
graph_data=dl.graph_data
dataset_train, dataset_test=graph_data()
print(dataset_train[0].x,dataset_train[0].a, dataset_train[0].y)
print(dataset_test[0].x,dataset_test[0].a, dataset_test[0].y)
print(len(dataset_train), len(dataset_test))