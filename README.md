# Investigating the expressive power of Graph Transformers
This repository contains code related to the master thesis 'Investigating the expressive power of Graph Transformers'.

It provides an implementation of the graphGPS layer by Rampášek et al. (https://arxiv.org/abs/2205.12454) 
and the SAN algorithm by Kreuzer et al. (https://arxiv.org/abs/2106.03893). 
# Code overview
The code is seperated into multiple folders containing files related to the respective parts of the architecture. 

/data/ contains the Code necessary for the implementation of the datasets and the encoding generating functions.

/layer/ contains the graphGPS architecture with the encoding, and layer architecture, as well as a Pytorch Geometric implementation of the SAN algorithm (Kreuzer et. al, 2021).


/models/ contains the graphGPS model file, combining the layer and encodings, as well as the configuration files for the different datasets.

/CSL/ contains code related to the evaluation of the CSL graphs.
slurm_basic.sh contains a simply slurm script to use the code on a slurm compatible cluster

# Usage guide

To use the graphGPS architecture with one of the implemented encodings, select the main file corresponding to the dataset that you want to evaluate.

Currently available models are: SAN_PE, SignNet_PE, RWPE_PE, RRWP_PE, SPE_PE and BasisNet_PE.

The selected encoding and the model architecture can be set in the config file with the parameters GT_dataset.node_encoder_name and GT_gt.layer_type.

The main files can be simply run using python <dataset>_main.py 

Furthermore, the configuration files provided in /models/config can be used to set the configuration values. An example file is given for the ZINC dataset and default values can be found in the config.py file. 
The scheduler and optimzer are set in the main files directly and are not supported via the configuration file at the moment.  





