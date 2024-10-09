from yacs.config import CfgNode as CN


def set_cfg(cfg):
    cfg.model = "GT"
    cfg.num_workers = 0
    cfg.device = "cuda"
    cfg.dataset = "ZINC"
    cfg.pre_train = False #pretraining for BREC
    cfg.dataset_node_emb = 1 #ZINC 28 for embedding, 1 for linear
    cfg.dataset_edge_emb = 1 #ZINC 4 for embedding, 1 for linear
    cfg.train_batch_size = 64 #16 for BREC
    cfg.test_batch_size = 64
    cfg.val_batch_size = 64
    cfg.train_epochs = 2000
    cfg.train_runs = 1
    cfg.seed = 41


    cfg.model_SAN = CN()
    cfg.model_SAN.gamma = 1e-5 #SAN gamma parameter
    cfg.model_SAN.num_atom_type = 28 #number of atomic types in the dataset 28 for ZINC
    cfg.model_SAN.num_bond_type = 4 #number of bond types present 4 for ZINC


    cfg.model_SAN.transformer_layers = 6 #number of transformer layers
    cfg.model_SAN.transformer_in_dim_nodes = 64 #input dimension of nodes
    cfg.model_SAN.transformer_in_dim_edges = 64 #input dimension of edges
    cfg.model_SAN.transformer_out_dim = 64 #output dimension of transformer
    cfg.model_SAN.transformer_hidden_dim = 64 #hidden dimension of transformer
    cfg.model_SAN.transformer_heads = 8 #number of transformer heads
    cfg.model_SAN.readout = "sum" #readout function
    cfg.model_SAN.residual = True #whether to use residual connections
    cfg.model_SAN.dropout =  0.0 #dropout probability
    cfg.model_SAN.layer_norm = False
    cfg.model_SAN.batch_norm = True
    cfg.model_SAN.lpe_dim = 16 #dimension of the learned encoding
    cfg.model_SAN.lpe_heads = 4 #number of transformer heads for the encoding
    cfg.model_SAN.lpe_layers = 3 #number of transformer layers for the encoding
    cfg.model_SAN.output_dim = 1 #output dimension of the algorithm


    cfg.BasisNet = CN()
    cfg.BasisNet.emb_dim = 16 #
    cfg.BasisNet.n_phis = 8 #number of phis used, currently supports 8 or 32
    cfg.BasisNet.node_feat = 1 #number of node features in data
    cfg.BasisNet.edge_feat = 1 #number of edge features in data
    cfg.BasisNet.node_out =16 #node embedding output dimension
    cfg.BasisNet.edge_out = 64 #edge embedding output dimension
    cfg.BasisNet.IGN_hidden_channel = 32 #IGN hidden channel
    cfg.BasisNet.n_layers = 4 #number of IGN layers

    cfg.GT_gnn = CN()

    cfg.GT_gnn.layers_pre_mp = 1 #pre mp layers
    cfg.GT_gnn.dim_inner = 64 #
    cfg.GT_gnn.act = "gelu" #activation function for GNN
    cfg.GT_gnn.agg = "mean" #aggregation function
    cfg.GT_gnn.dropout = 0


    cfg.GT_gt = CN()
    cfg.GT_gt.with_edges = True #using edge features
    cfg.GT_gt.in_dim = 40 #GT input dim
    cfg.GT_gt.layer_type = "GINE+Transformer" #type of layer
    cfg.GT_gt.layers = 8 #GT layers
    cfg.GT_gt.dim_hidden = 64 #GT hidden dimension
    cfg.GT_gt.n_heads = 4 #GT heads
    cfg.GT_gt.num_linear_emb_node = 1 #node embedding input dimension
    cfg.GT_gt.num_linear_emb_edge = 1 #node embedding input dimension

    cfg.GT_gt.pna_degrees = 0 #PNA degrees
    cfg.GT_gt.posenc_EquivStableLapPE = False #using EquivStablePE
    cfg.GT_gt.dropout = 0
    cfg.GT_gt.attn_dropout = 0.5
    cfg.GT_gt.layer_norm = False
    cfg.GT_gt.batch_norm = True

    cfg.GT_dataset = CN()
    cfg.GT_dataset.node_encoder = True
    cfg.GT_dataset.node_encoder_name = "BasisNet_PE"
    cfg.GT_dataset.node_encoder_bn = False
    cfg.GT_dataset.edge_encoder = False #no explicit edge encoder used
    cfg.GT_dataset.edge_encoder_name = None
    cfg.GT_dataset.edge_encoder_bn = False

    #config for each encoder (SAN, SignNet, SPE, RWPE) (node level)
    cfg.GT_encLAP = CN()
    cfg.GT_encLAP.embedding = "linear"
    cfg.GT_encLAP.dim_in = 40 #SAN input dimension transformer
    cfg.GT_encLAP.dim_pe = 16#encoding size
    cfg.GT_encLAP.layers = 3
    cfg.GT_encLAP.heads = 4
    cfg.GT_encLAP.post_layers = 0 #number of post transformer MLPs
    cfg.GT_encLAP.max_freq = 16 #number of eigenvalues/vectors used
    cfg.GT_encLAP.norm_type = "BatchNorm"

    cfg.GT_SignNetPE = CN()
    cfg.GT_SignNetPE.embedding = "linear"
    cfg.GT_SignNetPE.dim_in = 40
    cfg.GT_SignNetPE.dim_pe = 16
    cfg.GT_SignNetPE.model_type = "DeepSet"
    cfg.GT_SignNetPE.phi_layers = 8 #number of layers for phi
    cfg.GT_SignNetPE.rho_layers = 2 #number of layers for rho
    cfg.GT_SignNetPE.max_freq = 8 #number of eigenvectors used
    cfg.GT_SignNetPE.hidden_channel = 64 #number of hidden channels for phi, rho
    cfg.GT_SignNetPE.out_channel = 4 #number of output channels for phi, rho

    cfg.GT_SPE = CN()
    cfg.GT_SPE.embedding = "linear"
    cfg.GT_SPE.psi_layers = 3 #number of layers for psi
    cfg.GT_SPE.psi_hidden_dim = 16 #hidden dimension
    cfg.GT_SPE.psi_activation = "relu"
    cfg.GT_SPE.psi_batch_norm = "batch_norm"
    cfg.GT_SPE.n_psis = 8 #number of psis, equal to the number of eigenvectors used in the computation


    cfg.GT_SPE.phi_n_layers = 4 #phi layers
    cfg.GT_SPE.phi_hidden_dim = 128 #phi hidden dimension
    cfg.GT_SPE.phi_out_dim = 8 #phi output dimension
    cfg.GT_SPE.phi_batch_norm = "batch_norm"


    cfg.GT_SPE.MLP_hidden_dim = 128 #MLP dimension for SPE
    cfg.GT_SPE.MLP_n_layer = 3 #MLP layers
    cfg.GT_SPE.MLP_act = "relu"

    cfg.GT_RWPE =CN() #shares its parameters with GRIT
    cfg.GT_RWPE.embedding = "linear"
    cfg.GT_RWPE.dim_pe = 28 #encoding dimension
    cfg.GT_RWPE.steps = 16 #random walk steps
    cfg.GT_RWPE.model_type = "mlp"
    cfg.GT_RWPE.layers = 3 #number of MLP layers
    cfg.GT_RWPE.norm = "batchnorm"












    return cfg


import os
import argparse


"""
Update a config using the additional config files
"""
def update_config(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="", metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    # Update from command line
    cfg.merge_from_list(args.opts)

    return cfg


"""
    Global variable
"""
cfg = set_cfg(CN())