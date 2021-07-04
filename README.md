# IceCube
Collection of ML methods to analyze and reconstruct particle directions/energies as part of a bachelor project at the University of Copenhagen, handed in June 2021.

Since this project has been going on for a while, it has gotten a bit messy so here is a little guide to what to look for. Also please let me know if you find an error, it would be very appreciated!

## Structure

### Configuration style
The bulk of the project can be found in the folder from_config, which is structured after an idea from [Johann Bock Severin] (check his GitHub out!), where the idea is to be able to do training runs straight from a json-file containing everything relevant for the run. These can then be combined in a folder and all runs in the folder can be done. In here the basic structure is:

    "meta_data":{modelname/logging_method/...} 
    "run_params":{epochs/batch_size/patience/...}
    "hyper_params": {#conv_layers/#decode_layers/#hidden_states/...}
    "data_params":{database/graph_construction_method/...}

Just run run_trainings.py -f -gpu -cpus (sepcifying the folder where the experiment is to be found, the gpu, and the number of cpus to allocate), for the basic setup.

If the data is too large, then run run_submit.py, and specify how many chunks you want the data split up in with n_steps

I have further extended this to include uniform distribution sweeps, which can be done through sweep.py. Here you provide a base config file, and then a json with the different things you want to check out.

### Notebooks

My notebooks are messy, and as of the 5th of July, 2021, I have not cleaned them. I may make a more concise version of this repository later, but until then, venture into notebooks at your own risk.

### Other relevant information

The model architecture is called **StateFarm**, and I will publish a config for this such that others can retrain at will. Furthermore, I will publish the final trained model at different stages, such that others can play with it.

More about the fantastic group that most of the intellectual work can be found at https://www.nbi.dk/~petersen/IceCubeML/IceCubeML.html. Especially Jakob Schauser, Johann Bock Severin and Jonas Vinther are to thank for being fantastic groupmates, check out their work!

[Johann Bock Severin]:https://github.com/JohannSeverin/IceCube_GNN2
