import os.path

import torch
import torch_geometric
from layers.GT_layer import GT_layer, SAN_prediction_head
from layers.GT_encoder import SPEEncoder, LaplacianNodeEncoder, SignNetEncoder, RWPEEncoder, RRWPEncoder
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.register import register_network

from layers.BasisNetEncoder import BasisNetEncoder
@register_network('GPSModel')
class GT_model(torch.nn.Module):
    """
        Creates the GT model by using the configuration file. The model consists of an encoder, a pre_mp layer,
        transformer layers and an optional post_mp layer.
     """
    def __init__(self, config):
        super().__init__()

        #Creates the encoder used in the graphGPS architecture
        if config.GT_dataset.node_encoder:

            if config.GT_dataset.node_encoder_name == "SAN_PE":
                self.encoder = LaplacianNodeEncoder(config, dim_emb=config.GT_gt.in_dim)
            elif config.GT_dataset.node_encoder_name == "SignNet_PE":
                self.encoder = SignNetEncoder(config, dim_emb=config.GT_gt.in_dim)
            elif config.GT_dataset.node_encoder_name == "SPE_PE":
                self.encoder = SPEEncoder(config, dim_emb=config.GT_gt.in_dim)
            elif config.GT_dataset.node_encoder_name == "RWPE_PE":
                self.encoder = RWPEEncoder(config, dim_emb=config.GT_gt.in_dim)
            elif config.GT_dataset.node_encoder_name == "RRWP_PE":
                self.encoder = RRWPEncoder(config, dim_emb=config.GT_gt.in_dim)
            elif config.GT_dataset.node_encoder_name == "BasisNet_PE":
                self.encoder = BasisNetEncoder(config, dim_emb=config.GT_gt.in_dim)

        #Creating the pre_mp layer either by using an MLP or a GNN architecture
        if config.GT_gnn.layers_pre_mp > 0:

            self.pre_mp = torch_geometric.nn.MLP(in_channels=config.GT_gt.in_dim, hidden_channels=128, out_channels=config.GT_gnn.dim_inner, num_layers=config.GT_gnn.layers_pre_mp)
        else:
            self.pre_mp = torch.nn.Identity()
        #Get layer information from config
        try:
            local_gnn_type, global_model_type = config.GT_gt.layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {config.GT_gt.layer_type}")

        layers = []
        #Create graphGPS layer
        for _ in range(config.GT_gt.layers):
            layers.append(GT_layer(
                dim_h=config.GT_gt.dim_hidden,
                gnn_type=local_gnn_type,
                global_model=global_model_type,
                num_heads=config.GT_gt.n_heads,
                act=config.GT_gnn.act,
                pna_degrees=config.GT_gt.pna_degrees,
                equivstable_pe=config.GT_gt.posenc_EquivStableLapPE,
                dropout=config.GT_gt.dropout,
                attn_dropout=config.GT_gt.attn_dropout,
                layer_norm=config.GT_gt.layer_norm,
                batch_norm=config.GT_gt.batch_norm,
                log_attn_weights= 'log-attn-weights',
            ))
        self.layers = torch.nn.ModuleList(layers)

        #Create post_mp layer using the config file. The output dimension depends on the selected dataset.
        if config.dataset == "ALCHEMY":
            self.post_mp = SAN_prediction_head(config, config.GT_gnn.dim_inner, 12, L=2)
        elif config.dataset == "QM9":
            self.post_mp = SAN_prediction_head(config, config.GT_gnn.dim_inner, 19, L=2)
        elif config.dataset == "ZINC":
            self.post_mp = SAN_prediction_head(config, config.GT_gnn.dim_inner, 1, L=2)
        else:
            self.post_mp = SAN_prediction_head(config, config.GT_gnn.dim_inner, 1, L=2)


    def forward(self, data):
        """
        Applies the encoder, pre_mp, transformer and post_mp layer to the data object
        """
        data = self.encoder(data)

        data.x = self.pre_mp(data.x)

        for layer in self.layers:
            data = layer(data)

        h = self.post_mp(data)


        return h


    def reset_parameters(self):
        """
        Resets the parameters of each layer in the GT architecture
        """
        for layer in self.layers:
            layer.reset_parameters()
        self.post_mp.reset_parameters()
        self.pre_mp.reset_parameters()
        self.encoder.reset_parameters()
        return




